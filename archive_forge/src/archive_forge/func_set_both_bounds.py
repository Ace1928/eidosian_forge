import re
from . import constants as const
def set_both_bounds(value_low, value_up):
    set_one_bound('LO', value_low)
    set_one_bound('UP', value_up)