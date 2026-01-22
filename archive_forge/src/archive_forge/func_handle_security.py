from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def handle_security(controller, im_self=None):
    """ Checks the security of a controller.  """
    if controller._pecan.get('secured', False):
        check_permissions = controller._pecan['check_permissions']
        if isinstance(check_permissions, str):
            check_permissions = getattr(im_self or controller.__self__, check_permissions)
        if not check_permissions():
            raise exc.HTTPUnauthorized