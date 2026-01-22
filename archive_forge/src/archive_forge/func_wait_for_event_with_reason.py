import os
import random
import re
import subprocess
import time
import urllib
import fixtures
from heatclient import exc as heat_exceptions
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from tempest import config
import testscenarios
import testtools
from heat_integrationtests.common import clients
from heat_integrationtests.common import exceptions
def wait_for_event_with_reason(self, stack_identifier, reason, rsrc_name=None, num_expected=1):
    build_timeout = self.conf.build_timeout
    build_interval = self.conf.build_interval
    start = timeutils.utcnow()
    while timeutils.delta_seconds(start, timeutils.utcnow()) < build_timeout:
        try:
            rsrc_events = self.client.events.list(stack_identifier, resource_name=rsrc_name)
        except heat_exceptions.HTTPNotFound:
            LOG.debug('No events yet found for %s', rsrc_name)
        else:
            matched = [e for e in rsrc_events if e.resource_status_reason == reason]
            if len(matched) == num_expected:
                return matched
        time.sleep(build_interval)