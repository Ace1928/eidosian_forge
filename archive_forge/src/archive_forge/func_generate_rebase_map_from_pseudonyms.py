from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def generate_rebase_map_from_pseudonyms(pseudonym_dict, existing, desired):
    """Create a rebase map from pseudonyms and existing/desired ancestry.

    :param pseudonym_dict: Dictionary with pseudonym as returned by
        pseudonyms_as_dict()
    :param existing: Existing ancestry, might need to be rebased
    :param desired: Desired ancestry
    :return: rebase map, as dictionary
    """
    rebase_map = {}
    for revid in existing:
        for pn in pseudonym_dict.get(revid, []):
            if pn in desired:
                rebase_map[revid] = pn
    return rebase_map