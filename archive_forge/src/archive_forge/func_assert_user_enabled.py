import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def assert_user_enabled(self, user_id, user=None):
    """Assert the user and the user's domain are enabled.

        :raise AssertionError if the user or the user's domain is disabled.
        """
    if user is None:
        user = self.get_user(user_id)
    PROVIDERS.resource_api.assert_domain_enabled(user['domain_id'])
    if not user.get('enabled', True):
        raise AssertionError(_('User is disabled: %s') % user_id)