from numba.np.numpy_support import from_dtype
@cuda.jit(device=True)
def device_reduce_partial_block(arr, partials, sm_partials):
    """
        This computes reduction on `arr`.
        This device function must be used by 1 threadblock only.
        The blocksize must match `arr.size` and must not be greater than 128.
        """
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blksz = cuda.blockDim.x
    warpid = tid // _WARPSIZE
    laneid = tid % _WARPSIZE
    size = arr.size
    tid = cuda.threadIdx.x
    value = arr[tid]
    sm_partials[warpid, laneid] = value
    cuda.syncthreads()
    if (warpid + 1) * _WARPSIZE < size:
        inner_warp_reduction(sm_partials, value)
    elif laneid == 0:
        sm_this = sm_partials[warpid, :]
        base = warpid * _WARPSIZE
        for i in range(1, size - base):
            sm_this[0] = reduce_op(sm_this[0], sm_this[i])
    cuda.syncthreads()
    if tid == 0:
        num_active_warps = (blksz + _WARPSIZE - 1) // _WARPSIZE
        result = sm_partials[0, 0]
        for i in range(1, num_active_warps):
            result = reduce_op(result, sm_partials[i, 0])
        partials[blkid] = result