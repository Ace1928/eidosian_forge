import unittest
@njit
def inside_interval(interval, x):
    return interval.lo <= x < interval.hi