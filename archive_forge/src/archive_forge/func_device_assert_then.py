import triton
import triton.language as tl
@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r