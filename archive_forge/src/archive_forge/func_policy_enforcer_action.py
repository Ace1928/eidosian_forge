import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def policy_enforcer_action(cls, action):
    """Decorator to set policy enforcement action name."""
    if action not in _POSSIBLE_TARGET_ACTIONS:
        raise ValueError('PROGRAMMING ERROR: Action must reference a valid Keystone policy enforcement name.')

    def wrapper(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):
            setattr(flask.g, cls.ACTION_STORE_ATTR, action)
            return f(*args, **kwargs)
        return inner
    return wrapper