import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_update_with_adjusted_capacity(self):
    """Test rolling update with capacity adjustment.

        Rolling update with capacity adjustment due to conflict in
        batch size and minimum instances in service.
        """
    updt_template = yaml.safe_load(copy.deepcopy(self.template))
    grp = updt_template['resources']['random_group']
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '8'
    policy['max_batch_size'] = '4'
    res_def = grp['properties']['resource_def']
    res_def['properties']['value'] = 'updated'
    self.update_resource_group(updt_template, updated=10, created=0, deleted=0)