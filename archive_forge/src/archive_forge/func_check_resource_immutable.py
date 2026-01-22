from keystone.common.resource_options import core as ro_core
from keystone.common.validation import parameter_types
from keystone import exception
def check_resource_immutable(resource_ref):
    """Check to see if a resource is immutable.

    :param resource_ref: a dict reference of a resource to inspect
    """
    return resource_ref.get('options', {}).get(IMMUTABLE_OPT.option_name, False)