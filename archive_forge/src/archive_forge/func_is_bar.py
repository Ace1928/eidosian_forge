import math
import warnings
import matplotlib.dates
def is_bar(bar_containers, **props):
    """A test to decide whether a path is a bar from a vertical bar chart."""
    for container in bar_containers:
        if props['mplobj'] in container:
            return True
    return False