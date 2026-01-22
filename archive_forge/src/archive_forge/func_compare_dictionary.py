from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
def compare_dictionary(want, have):
    """Performs a dictionary comparison

    Args:
        want (dict): Dictionary to compare with second parameter.
        have (dict): Dictionary to compare with first parameter.

    Returns:
        bool:
    """
    if want == {} and have is None:
        return None
    if want is None:
        return None
    w = [(str(k), str(v)) for k, v in iteritems(want)]
    h = [(str(k), str(v)) for k, v in iteritems(have)]
    if set(w) == set(h):
        return None
    else:
        return want