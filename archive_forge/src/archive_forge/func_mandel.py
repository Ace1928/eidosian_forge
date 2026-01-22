from numba import float64, uint32
from numba.cuda.compiler import compile_ptx
from numba.cuda.testing import skip_on_cudasim, unittest
def mandel(tid, min_x, max_x, min_y, max_y, width, height, iters):
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    x = tid % width
    y = tid / width
    real = min_x + x * pixel_size_x
    imag = min_y + y * pixel_size_y
    c = complex(real, imag)
    z = 0j
    for i in range(iters):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return iters