from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def parse_mappings(mapping_list, unique_values=True, unique_keys=True):
    """Parse a list of mapping strings into a dictionary.

    :param mapping_list: A list of strings of the form '<key>:<value>'.
    :param unique_values: Values must be unique if True.
    :param unique_keys: Keys must be unique if True, else implies that keys
        and values are not unique.
    :returns: A dict mapping keys to values or to list of values.
    :raises ValueError: Upon malformed data or duplicate keys.
    """
    mappings = {}
    for mapping in mapping_list:
        mapping = mapping.strip()
        if not mapping:
            continue
        split_result = mapping.split(':')
        if len(split_result) != 2:
            raise ValueError(_("Invalid mapping: '%s'") % mapping)
        key = split_result[0].strip()
        if not key:
            raise ValueError(_("Missing key in mapping: '%s'") % mapping)
        value = split_result[1].strip()
        if not value:
            raise ValueError(_("Missing value in mapping: '%s'") % mapping)
        if unique_keys:
            if key in mappings:
                raise ValueError(_("Key %(key)s in mapping: '%(mapping)s' not unique") % {'key': key, 'mapping': mapping})
            if unique_values and value in mappings.values():
                raise ValueError(_("Value %(value)s in mapping: '%(mapping)s' not unique") % {'value': value, 'mapping': mapping})
            mappings[key] = value
        else:
            mappings.setdefault(key, [])
            if value not in mappings[key]:
                mappings[key].append(value)
    return mappings