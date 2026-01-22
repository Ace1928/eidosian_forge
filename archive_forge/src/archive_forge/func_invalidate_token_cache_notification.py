import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def invalidate_token_cache_notification(reason):
    """A specific notification for invalidating the token cache.

    :param reason: The specific reason why the token cache is being
                   invalidated.
    :type reason: string

    """
    LOG.debug(reason)
    resource_id = None
    initiator = None
    public = False
    Audit._emit(ACTIONS.internal, INVALIDATE_TOKEN_CACHE, resource_id, initiator, public, reason=reason)