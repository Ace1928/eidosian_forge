from numba import typed, int64
def global_reflected_write():
    _global_list[0] = 10