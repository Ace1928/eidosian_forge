from numba.cuda.cudadrv import devices, driver
from numba.core.registry import cpu_target
def ndarray_device_allocate_data(ary):
    """
    Allocate gpu data buffer
    """
    datasize = driver.host_memory_size(ary)
    gpu_data = devices.get_context().memalloc(datasize)
    return gpu_data