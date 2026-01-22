import netaddr
import re
from heat.common.i18n import _
def is_prefix_subset(orig_prefixes, new_prefixes):
    """Check whether orig_prefixes is subset of new_prefixes.


    This takes valid prefix lists for orig_prefixes and new_prefixes,
    returns 'True', if orig_prefixes is subset of new_prefixes.
    """
    orig_set = netaddr.IPSet(orig_prefixes)
    new_set = netaddr.IPSet(new_prefixes)
    return orig_set.issubset(new_set)