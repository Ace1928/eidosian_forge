import triton
import triton.language as tl
@triton.jit
def promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)