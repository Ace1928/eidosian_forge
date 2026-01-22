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
def assert_resource_is_a_stack(self, stack_identifier, res_name, wait=False):
    build_timeout = self.conf.build_timeout
    build_interval = self.conf.build_interval
    start = timeutils.utcnow()
    while timeutils.delta_seconds(start, timeutils.utcnow()) < build_timeout:
        time.sleep(build_interval)
        try:
            nested_identifier = self._get_nested_identifier(stack_identifier, res_name)
        except Exception:
            if wait:
                time.sleep(build_interval)
            else:
                raise
        else:
            return nested_identifier