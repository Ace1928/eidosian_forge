import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def generate_patches_from_kwargs(operation, **kwargs):
    """Given a set of parameters, returns a list with the
    valid patch values.

    :param string operation: The operation to perform.
    :param list kwargs: Dict of parameters.

    :returns: A list with the right patch values.
    """
    patches = []
    for k, v in kwargs.items():
        patch = {'op': operation, 'value': v, 'path': '/%s' % k}
        patches.append(patch)
    return sorted(patches)