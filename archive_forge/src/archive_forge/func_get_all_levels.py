import sys
import logging
def get_all_levels():
    levels = set(logging._nameToLevel.keys())
    levels.remove('WARNING')
    return [l.lower() for l in levels]