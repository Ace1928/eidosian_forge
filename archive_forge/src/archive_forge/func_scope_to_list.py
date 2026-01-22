from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def scope_to_list(scope):
    """Convert a space separated string to a list of scopes."""
    if isinstance(scope, (tuple, list, set)):
        return [unicode_type(s) for s in scope]
    elif scope is None:
        return None
    else:
        return scope.strip().split(' ')