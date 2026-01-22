import logging
import re
def to_comparators(range_, loose):
    return [' '.join([c.value for c in comp]).strip().split(' ') for comp in make_range(range_, loose).set]