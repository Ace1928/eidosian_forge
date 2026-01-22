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
def check_input_values(self, group_resources, key, value):
    for r in group_resources:
        d = self.client.software_deployments.get(r.physical_resource_id)
        self.assertEqual({key: value}, d.input_values)
        c = self.client.software_configs.get(d.config_id)
        foo_input_c = [i for i in c.inputs if i.get('name') == key][0]
        self.assertEqual(value, foo_input_c.get('value'))