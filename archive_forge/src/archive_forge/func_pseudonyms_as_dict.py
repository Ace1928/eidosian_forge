from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def pseudonyms_as_dict(l):
    """Convert an iterable over pseudonyms to a dictionary.

    :param l: Iterable over sets of pseudonyms
    :return: Dictionary with pseudonyms for each revid.
    """
    ret = {}
    for pns in l:
        for pn in pns:
            ret[pn] = pns - {pn}
    return ret