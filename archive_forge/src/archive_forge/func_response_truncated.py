import functools
import inspect
import time
import types
from oslo_log import log
import stevedore
from keystone.common import provider_api
from keystone.i18n import _
def response_truncated(f):
    """Truncate the list returned by the wrapped function.

    This is designed to wrap Manager list_{entity} methods to ensure that
    any list limits that are defined are passed to the driver layer.  If a
    hints list is provided, the wrapper will insert the relevant limit into
    the hints so that the underlying driver call can try and honor it. If the
    driver does truncate the response, it will update the 'truncated' attribute
    in the 'limit' entry in the hints list, which enables the caller of this
    function to know if truncation has taken place.  If, however, the driver
    layer is unable to perform truncation, the 'limit' entry is simply left in
    the hints list for the caller to handle.

    A _get_list_limit() method is required to be present in the object class
    hierarchy, which returns the limit for this backend to which we will
    truncate.

    If a hints list is not provided in the arguments of the wrapped call then
    any limits set in the config file are ignored.  This allows internal use
    of such wrapped methods where the entire data set is needed as input for
    the calculations of some other API (e.g. get role assignments for a given
    project).

    """

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if kwargs.get('hints') is None:
            return f(self, *args, **kwargs)
        list_limit = self.driver._get_list_limit()
        if list_limit:
            kwargs['hints'].set_limit(list_limit)
        return f(self, *args, **kwargs)
    return wrapper