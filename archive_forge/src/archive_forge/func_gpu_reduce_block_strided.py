from numba.np.numpy_support import from_dtype
def gpu_reduce_block_strided(arr, partials, init, use_init):
    """
        Perform reductions on *arr* and writing out partial reduction result
        into *partials*.  The length of *partials* is determined by the
        number of threadblocks. The initial value is set with *init*.

        Launch config:

        Blocksize must be multiple of warpsize and it is limited to 4 warps.
        """
    tid = cuda.threadIdx.x
    sm_partials = cuda.shared.array((_NUMWARPS, inner_sm_size), dtype=nbtype)
    if cuda.blockDim.x == max_blocksize:
        device_reduce_full_block(arr, partials, sm_partials)
    else:
        device_reduce_partial_block(arr, partials, sm_partials)
    if use_init and tid == 0 and (cuda.blockIdx.x == 0):
        partials[0] = reduce_op(partials[0], init)