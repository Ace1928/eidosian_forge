import builtins
import sys
def make_identity_dict(rng):
    """ make_identity_dict(rng) -> dict

        Return a dictionary where elements of the rng sequence are
        mapped to themselves.

    """
    return {i: i for i in rng}