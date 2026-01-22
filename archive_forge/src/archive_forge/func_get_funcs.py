import collections
import inspect
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.utils import helpers
def get_funcs(resource):
    """Retrieve a list of functions extending a resource.

    :param resource: A resource collection name.
    :type resource: str

    :return: A list (possibly empty) of functions extending resource.
    :rtype: list of callable

    """
    return _resource_extend_functions.get(resource, [])