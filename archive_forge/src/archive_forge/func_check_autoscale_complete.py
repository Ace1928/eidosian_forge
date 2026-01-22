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
def check_autoscale_complete(self, stack_id, expected_num, parent_stack, group_name):
    res_list = self.client.resources.list(stack_id)
    all_res_complete = all((res.resource_status in ('UPDATE_COMPLETE', 'CREATE_COMPLETE') for res in res_list))
    all_res = len(res_list) == expected_num
    if all_res and all_res_complete:
        metadata = self.client.resources.metadata(parent_stack, group_name)
        return not metadata.get('scaling_in_progress')
    return False