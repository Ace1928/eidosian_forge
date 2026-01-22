import collections
import inspect
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.utils import helpers
Decorator to setup __new__ method in classes to extend resources.

    Any method decorated with @extends above is an unbound method on a class.
    This decorator sets up the class __new__ method to add the bound
    method to _resource_extend_functions after object instantiation.
    