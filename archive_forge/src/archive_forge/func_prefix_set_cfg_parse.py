from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.prefix_lists.prefix_lists import Prefix_listsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def prefix_set_cfg_parse(unparsed_prefix_set):
    """Parse the raw input configuration JSON representation for the prefix set specified
    by the input "unparsed_prefix_set" input parameter. Parse the information to
    convert it to a dictionary matching the "argspec" for the "prefix_lists" resource
    module."""
    parsed_prefix_set = dict()
    if not unparsed_prefix_set.get('config'):
        return parsed_prefix_set
    parsed_prefix_set['name'] = unparsed_prefix_set['name']
    pfx_cfg = unparsed_prefix_set['config']
    if pfx_cfg.get('mode') and isinstance(pfx_cfg['mode'], str):
        parsed_prefix_set['afi'] = pfx_cfg['mode'].lower()
    if unparsed_prefix_set.get('openconfig-routing-policy-ext:extended-prefixes'):
        prefix_lists_container = unparsed_prefix_set['openconfig-routing-policy-ext:extended-prefixes']
        if not prefix_lists_container.get('extended-prefix'):
            return parsed_prefix_set
        prefix_lists_unparsed = prefix_lists_container['extended-prefix']
        prefix_lists_parsed = []
        for prefix_entry_unparsed in prefix_lists_unparsed:
            if not prefix_entry_unparsed.get('config'):
                continue
            if not prefix_entry_unparsed['config'].get('action'):
                continue
            prefix_entry_cfg = prefix_entry_unparsed['config']
            prefix_parsed = dict()
            prefix_parsed['action'] = prefix_entry_cfg['action'].lower()
            if not prefix_entry_unparsed.get('ip-prefix'):
                continue
            if not prefix_entry_unparsed.get('sequence-number'):
                continue
            prefix_parsed['prefix'] = prefix_entry_unparsed['ip-prefix']
            prefix_parsed['sequence'] = prefix_entry_unparsed['sequence-number']
            if prefix_entry_unparsed.get('masklength-range') and (not prefix_entry_unparsed['masklength-range'] == 'exact'):
                mask = int(prefix_parsed['prefix'].split('/')[1])
                ge_le = prefix_entry_unparsed['masklength-range'].split('..')
                ge_bound = int(ge_le[0])
                if ge_bound != mask:
                    prefix_parsed['ge'] = ge_bound
                pfx_len = 32 if parsed_prefix_set['afi'] == 'ipv4' else 128
                le_bound = int(ge_le[1])
                if le_bound != pfx_len:
                    prefix_parsed['le'] = le_bound
            prefix_lists_parsed.append(prefix_parsed)
        parsed_prefix_set['prefixes'] = prefix_lists_parsed
    return parsed_prefix_set