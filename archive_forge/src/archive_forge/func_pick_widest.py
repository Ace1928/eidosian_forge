from taskflow.utils import misc
def pick_widest(depths):
    """Pick from many depths which has the **widest** area of influence."""
    return _ORDERING[min((_ORDERING.index(d) for d in depths))]