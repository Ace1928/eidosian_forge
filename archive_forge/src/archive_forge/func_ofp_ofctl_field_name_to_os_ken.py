import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
def ofp_ofctl_field_name_to_os_ken(field):
    """Convert an ovs-ofctl field name to os_ken equivalent."""
    mapped = _OXM_FIELD_OFCTL_ALIASES.get(field)
    if mapped:
        return mapped
    if field.endswith('_dst'):
        mapped = _OXM_FIELD_OFCTL_ALIASES.get(field[:-3] + 'src')
        if mapped:
            return mapped[:-3] + 'dst'
    return field