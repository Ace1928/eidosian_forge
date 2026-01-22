import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_instance_group_update_no_replace(self):
    """Test simple update only and no replace with no conflict.

        Test simple update only and no replace (i.e. updated instance flavor
        in Launch Configuration) with no conflict in batch size and
        minimum instances in service.
        """
    updt_template = self.ig_tmpl_with_updt_policy()
    group = updt_template['Resources']['JobServerGroup']
    policy = group['UpdatePolicy']['RollingUpdate']
    policy['MinInstancesInService'] = '1'
    policy['MaxBatchSize'] = '3'
    policy['PauseTime'] = 'PT0S'
    config = updt_template['Resources']['JobServerConfig']
    config['Properties']['InstanceType'] = self.conf.instance_type
    self.update_instance_group(updt_template, num_updates_expected_on_updt=5, num_creates_expected_on_updt=0, num_deletes_expected_on_updt=0, update_replace=False)