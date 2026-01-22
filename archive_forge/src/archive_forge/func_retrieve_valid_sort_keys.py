from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def retrieve_valid_sort_keys(attr_info):
    """Retrieve sort keys from `attr_info` dict.

    Iterate the `attr_info`, filter and return the attributes that are
    defined with `is_sort_key=True`.

    :param attr_info: The attribute dict for common neutron resource.
    :returns: Set of sort keys.
    """
    return set((attr for attr, schema in attr_info.items() if schema.get('is_sort_key', False)))