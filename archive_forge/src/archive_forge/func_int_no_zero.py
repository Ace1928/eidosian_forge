from copy import deepcopy
def int_no_zero(val):
    """Return integer of val, or None if is zero."""
    v = int(val)
    if v == 0:
        return None
    return v