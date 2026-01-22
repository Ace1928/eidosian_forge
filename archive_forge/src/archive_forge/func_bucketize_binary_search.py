import triton
import triton.language as tl
@triton.jit
def bucketize_binary_search(values, offsets_ptr, indexing_dtype, right, OFFSETS_SIZE: int, BLOCK_SHAPE):
    """
    See [Note: Inductor bucketize op]
    """
    low = tl.zeros(BLOCK_SHAPE, dtype=indexing_dtype)
    high = tl.full(BLOCK_SHAPE, OFFSETS_SIZE, dtype=indexing_dtype)
    full_range = OFFSETS_SIZE + 1
    while full_range > 1:
        mid = (high + low) // 2
        mask = mid < OFFSETS_SIZE
        bucket_upper_bound = tl.load(offsets_ptr + mid, mask=mask)
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound
        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)
        full_range = (full_range + 1) // 2
    return low