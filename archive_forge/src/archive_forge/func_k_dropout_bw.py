import triton
import triton.language as tl
from xformers.triton.k_activations import (
@triton.heuristics({'SIZE_RAND_BLOCK': lambda args: args['BLOCK_N'] * args['BLOCK_M']})
@triton.autotune(configs=_configs, key=['M', 'N', 'is_fp16'])
@triton.jit
def k_dropout_bw(GRAD_IN, GRAD_BIAS, GRAD_OUT, INPUTS, BIAS, SEEDS, stride_grad, stride_inputs, M, N, p: tl.constexpr, is_fp16: tl.constexpr, ACTIVATION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SIZE_RAND_BLOCK: tl.constexpr, TRAINABLE_BIAS: tl.constexpr, USE_BIAS: tl.constexpr):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_BIAS   (N,)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (N,)
    p : dropout probability
    """
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    grad_out_ptrs = GRAD_OUT + rows[:, None] * stride_grad + cols[None, :]
    grad_in_ptrs = GRAD_IN + rows[:, None] * stride_grad + cols[None, :]
    input_ptrs = INPUTS + rows[:, None] * stride_inputs + cols[None, :]
    grad_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    col_mask = cols[None, :] < N
    p_scale = 1.0 / (1.0 - p)
    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=col_mask, other=0.0)
    block_mask = (rows[:, None] < M) & col_mask
    grad_out = tl.load(grad_out_ptrs, mask=block_mask, other=0.0)
    if ACTIVATION:
        inputs = tl.load(input_ptrs, mask=block_mask, other=0.0)
        if USE_BIAS:
            inputs += bias
        if ACTIVATION == 1:
            act_grad = relu_grad(inputs)
        elif ACTIVATION == 2:
            act_grad = leaky_relu_grad(inputs)
        elif ACTIVATION == 3:
            act_grad = gelu_grad(inputs)
        elif ACTIVATION == 4:
            act_grad = squared_relu_grad(inputs)
        elif ACTIVATION == 5:
            act_grad = smelu_grad(inputs)
        grad_out *= act_grad
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
    seed_int = tl.load(SEEDS + col_id)
    r = tl.rand(seed_int, rand_offsets)
    r = tl.view(r, grad_out.shape)
    output = tl.where(r > p, (grad_out * p_scale).to(grad_out.dtype), 0.0)
    tl.store(grad_in_ptrs, output, mask=block_mask)
    if TRAINABLE_BIAS:
        grad_bias += tl.sum(output, axis=0)
    if TRAINABLE_BIAS:
        grad_bias_ptr = GRAD_BIAS + row_id * N + cols
        tl.store(grad_bias_ptr, grad_bias, mask=cols < N)