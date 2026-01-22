import numpy as np
def get_best_unit(unit_a, unit_b):
    """
    Get the best (i.e. finer-grained) of two units.
    """
    a = DATETIME_UNITS[unit_a]
    b = DATETIME_UNITS[unit_b]
    if a == 14:
        return unit_b
    if b == 14:
        return unit_a
    if b > a:
        return unit_b
    return unit_a