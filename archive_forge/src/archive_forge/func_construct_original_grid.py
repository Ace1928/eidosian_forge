import numpy as np
from onnx.reference.op_run import OpRun
def construct_original_grid(data_size, align_corners):
    is_2d = len(data_size) == 2
    size_zeros = np.zeros(data_size)
    original_grid = [np.ones(data_size)]
    for dim, dim_size in enumerate(data_size):
        if align_corners == 1:
            step = 2.0 / (dim_size - 1)
            start = -1
            stop = 1 + 0.0001
            a = np.arange(start, stop, step)
        else:
            step = 2.0 / dim_size
            start = -1 + step / 2
            stop = 1
            a = np.arange(start, stop, step)
        if dim == 0:
            if is_2d:
                y = np.reshape(a, (dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
            else:
                z = np.reshape(a, (dim_size, 1, 1)) + size_zeros
                original_grid = [z, *original_grid]
        elif dim == 1:
            if is_2d:
                x = np.reshape(a, (1, dim_size)) + size_zeros
                original_grid = [x, *original_grid]
            else:
                y = np.reshape(a, (1, dim_size, 1)) + size_zeros
                original_grid = [y, *original_grid]
        else:
            x = np.reshape(a, (1, dim_size)) + size_zeros
            original_grid = [x, *original_grid]
    return np.stack(original_grid, axis=2 if is_2d else 3)