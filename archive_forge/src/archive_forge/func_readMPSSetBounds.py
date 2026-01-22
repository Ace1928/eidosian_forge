import re
from . import constants as const
def readMPSSetBounds(line, variable_dict):
    bound = line[0]
    var_name = line[2]

    def set_one_bound(bound_type, value):
        variable_dict[var_name][BOUNDS_EQUIV[bound_type]] = value

    def set_both_bounds(value_low, value_up):
        set_one_bound('LO', value_low)
        set_one_bound('UP', value_up)
    if bound == 'FR':
        set_both_bounds(None, None)
        return
    elif bound == 'BV':
        set_both_bounds(0, 1)
        return
    value = float(line[3])
    if bound in ['LO', 'UP']:
        set_one_bound(bound, value)
    elif bound == 'FX':
        set_both_bounds(value, value)
    return