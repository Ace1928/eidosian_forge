from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def memory_efficient_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[Union[torch.Tensor, AttentionBias]]=None, p: float=0.0, scale: Optional[float]=None, *, op: Optional[AttentionOp]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    """Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    :Inputs shape:

    - Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M         the sequence length, H the number of heads, and K the embeding size per head

    - If inputs have dimension 3, it is assumed that the dimensions are ``[B, M, K]`` and ``H=1``

    - Inputs can also be of dimension 5 with GQA - see note below

    - Inputs can be non-contiguous - we only require the last dimension's stride to be 1


    :Equivalent pytorch code:

    .. code-block:: python

        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        attn = attn @ value
        return attn.transpose(1, 2)

    :Examples:

    .. code-block:: python

        import xformers.ops as xops

        # Compute regular attention
        y = xops.memory_efficient_attention(q, k, v)

        # With a dropout of 0.2
        y = xops.memory_efficient_attention(q, k, v, p=0.2)

        # Causal attention
        y = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask()
        )

    :Supported hardware:

        NVIDIA GPUs with compute capability above 6.0 (P100+), datatype ``f16``, ``bf16`` and ``f32``.

    :EXPERIMENTAL: Using with Multi Query Attention (MQA) and Grouped Query Attention (GQA):

        MQA/GQA is an experimental feature supported only for the forward pass.
        If you have 16 heads in query, and 2 in key/value, you can provide 5-dim tensors
        in the ``[B, M, G, H, K]`` format, where ``G`` is the number of head groups (here 2), and
        ``H`` is the number of heads per group (8 in the example).

        Please note that xFormers will not automatically broadcast the inputs, so you will need
        to broadcast it manually before calling `memory_efficient_attention`.

    :GQA/MQA example:

    .. code-block:: python

        import torch
        import xformers.ops as xops

        B, M, K = 3, 32, 128
        kwargs = dict(device="cuda", dtype=torch.float16)
        q = torch.randn([B, M, 8, K], **kwargs)
        k = torch.randn([B, M, 2, K], **kwargs)
        v = torch.randn([B, M, 2, K], **kwargs)
        out_gqa = xops.memory_efficient_attention(
            q.reshape([B, M, 2, 4, K]),
            k.reshape([B, M, 2, 1, K]).expand([B, M, 2, 4, K]),
            v.reshape([B, M, 2, 1, K]).expand([B, M, 2, 4, K]),
        )

    Raises:
        NotImplementedError: if there is no operator available to compute the MHA
        ValueError: if inputs are invalid

    :parameter query: Tensor of shape ``[B, Mq, H, K]``
    :parameter key: Tensor of shape ``[B, Mkv, H, K]``
    :parameter value: Tensor of shape ``[B, Mkv, H, Kv]``
    :parameter attn_bias: Bias to apply to the attention matrix - defaults to no masking.         For common biases implemented efficiently in xFormers, see :attr:`xformers.ops.fmha.attn_bias.AttentionBias`.         This can also be a :attr:`torch.Tensor` for an arbitrary mask (slower).
    :parameter p: Dropout probability. Disabled if set to ``0.0``
    :parameter scale: Scaling factor for ``Q @ K.transpose()``. If set to ``None``, the default         scale (q.shape[-1]**-0.5) will be used.
    :parameter op: The operators to use - see :attr:`xformers.ops.AttentionOpBase`.         If set to ``None`` (recommended), xFormers         will dispatch to the best available operator, depending on the inputs         and options.
    :return: multi-head attention Tensor with shape ``[B, Mq, H, Kv]``
    """
    return _memory_efficient_attention(Inputs(query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale, output_dtype=output_dtype), op=op)