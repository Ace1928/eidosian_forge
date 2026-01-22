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
def check_config_and_reload_domain_driver_if_required(self, domain_id):
    """Check for, and load, any new domain specific config for this domain.

        This is only supported for the database-stored domain specific
        configuration.

        When the domain specific drivers were set up, we stored away the
        specific config for this domain that was available at that time. So we
        now read the current version and compare. While this might seem
        somewhat inefficient, the sensitive config call is cached, so should be
        light weight. More importantly, when the cache timeout is reached, we
        will get any config that has been updated from any other keystone
        process.

        This cache-timeout approach works for both multi-process and
        multi-threaded keystone configurations. In multi-threaded
        configurations, even though we might remove a driver object (that
        could be in use by another thread), this won't actually be thrown away
        until all references to it have been broken. When that other
        thread is released back and is restarted with another command to
        process, next time it accesses the driver it will pickup the new one.

        """
    if not CONF.identity.domain_specific_drivers_enabled or not CONF.identity.domain_configurations_from_database:
        return
    latest_domain_config = PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain_id)
    domain_config_in_use = domain_id in self
    if latest_domain_config:
        if not domain_config_in_use or latest_domain_config != self[domain_id]['cfg_overrides']:
            self._load_config_from_database(domain_id, latest_domain_config)
    elif domain_config_in_use:
        try:
            del self[domain_id]
        except KeyError:
            pass