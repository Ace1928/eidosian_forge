import logging
import re
def make_range(range_, loose):
    if isinstance(range_, Range) and range_.loose == loose:
        return range_
    return Range(range_, loose)