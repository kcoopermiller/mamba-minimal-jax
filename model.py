"""Reimplemntation of Mamba-minimal in JAX/Flax.

Refer to the orginal PyTorch Mamba-minimal or the Mamba paper/implementation for more details:
https://github.com/johnma2006/mamba-minimal
https://github.com/state-spaces/mamba
"""
from __future__ import annotations
import math
import json
import torch
from jax import random
import jax.numpy as jnp
from flax import traverse_util
from flax import linen as nn
from dataclasses import dataclass
from typing import Union, Dict

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    args: ModelArgs

    def setup(self):
        """Full Mamba model."""
        super().__init__()
    
        self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_f = nn.RMSNorm(epsilon=1e-5)

        self.lm_head = lambda input: self.embedding.attend(input)  # weight tying
    
    @nn.compact
    def __call__(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits


    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            """Fetches the model configuration (like number of layers, model dimension, vocabulary size) from the HuggingFace model hub."""
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            """Fetches the pretrained model weights from the HuggingFace model hub."""
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location=torch.device('cpu'))
        
        def pt_to_jax(state_dict: Dict, params: Dict):
            """Converts a PyTorch state dictionary to a format compatible with a JAX model.
            Inspired by https://github.com/radarFudan/mamba-minimal-jax/blob/main/pytorch_to_jax.py

            Args:
                state_dict (Dict): The state dictionary from a PyTorch model.
                params (Dict): The parameter dictionary of a JAX model.

            Returns:
                Dict: Converted parameters compatible with JAX.
            """
            def convert_key(pytorch_key: str) -> str:
                """Converts PyTorch tensor keys to JAX-compatible keys."""
                replacements = {
                    "embedding.weight": "embedding.embedding",
                    "layers.": "layers_",
                    "proj.weight": "proj.kernel",
                    "conv1d.weight": "conv1d.kernel",
                    "norm_f.weight": "norm_f.scale",
                    "norm.weight": "norm.scale"
                }
                for old, new in replacements.items():
                    pytorch_key = pytorch_key.replace(old, new)
                return pytorch_key
            
            params = traverse_util.flatten_dict(params, sep=".")
            jax_state = {}
            for key, tensor in state_dict.items():
                new_key = convert_key(key)

                if new_key in params:
                    tensor_np = tensor.cpu().numpy()
                    if tensor_np.shape != params[new_key].shape:
                        tensor_np = tensor_np.T
                    jax_state[new_key] = tensor_np

            # Validate shapes and data types
            for key in jax_state:
                assert jax_state[key].shape == params[key].shape, \
                    f'Shape mismatch for {key}: JAX {params[key].shape}, converted {jax_state[key].shape}'
                assert jax_state[key].dtype == params[key].dtype, \
                    f'Dtype mismatch for {key}: JAX {params[key].dtype}, converted {jax_state[key].dtype}'

            # Reconstruct the JAX parameter dictionary
            return traverse_util.unflatten_dict(jax_state, sep=".")


        # ModelArgs created with parameters extracted from config_data.
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )

        # Mamba model is initialized with the ModelArgs.
        model = Mamba(args)
    
        state_dict = load_state_dict_hf(pretrained_model_name)
        # Convert the state_dict to match the Flax model.
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', 'params.')
            new_state_dict[new_key] = state_dict[key]
        
        # Initialize the model with a dummy input.
        rng = random.PRNGKey(7)
        params = model.init(rng, jnp.zeros((1, 1), dtype=jnp.int32))
        params = pt_to_jax(new_state_dict, params)
        
        return model, params


class ResidualBlock(nn.Module):
    args:ModelArgs

    def setup(self):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = MambaBlock(self.args)
        self.norm = nn.RMSNorm(epsilon=1e-5)

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        self.in_proj = nn.Dense(features=self.args.d_inner * 2, use_bias=self.args.bias)
        
        self.conv1d = nn.Conv(
            features=self.args.d_inner,
            use_bias=self.args.conv_bias,
            kernel_size=[self.args.d_conv],
            feature_group_count=self.args.d_inner,
            padding=self.args.d_conv - 1
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)

        A = jnp.tile(jnp.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param('A_log', lambda rng, shape: jnp.log(A), A.shape)
        self.D = self.param('D', nn.initializers.ones, (self.args.d_inner,))
        self.out_proj = nn.Dense(self.args.d_model, use_bias=self.args.bias)


    def __call__(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
 
        (x, res) = jnp.split(x_and_res, indices_or_sections=[self.args.d_inner,], axis=-1)

        # TODO I think this is a correct explanation for the difference between the PyTorch and Flax implementations???
        # PyTorch typically uses the 'channels-first' convention for convolutional layers:
        # [batch_size, channels, length] 
        # So, in the PyTorch implementation of the Mamba-minimal, you see the input x being rearranged to fit this convention before applying the conv1d operation:
        # x = rearrange(x, 'b l d_in -> b d_in l')
        # x = self.conv1d(x)[:, :, :l]
        # x = rearrange(x, 'b d_in l -> b l d_in')
        # In Flax, we use the channels last convention [batch_size, length, channels], so we don't need to rearrange the input
        x = self.conv1d(x)[:, :l, :]

        x = nn.silu(x)

        y = self.ssm(x)
        
        y = y * nn.silu(res)
        
        output = self.out_proj(y)

        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -jnp.exp(self.A_log)  # shape (d_in, n)
        D = self.D.astype(jnp.float32)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = jnp.split(x_dbl, indices_or_sections=[self.args.dt_rank, self.args.dt_rank + n], axis=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = nn.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = jnp.exp(jnp.einsum('b l d, d n -> b l d n', delta, A))
        deltaB_u = jnp.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = jnp.zeros((b, d_in, n))
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = jnp.einsum('b d n, b n -> b d', x, C[:, i, :])
            ys.append(y)
        y = jnp.stack(ys, axis=1)  # shape (b, l, d_in)
        
        y = y + u * D

        return y
