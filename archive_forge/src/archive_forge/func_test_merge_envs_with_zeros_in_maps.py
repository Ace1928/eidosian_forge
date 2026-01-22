import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_envs_with_zeros_in_maps(self):
    env1 = {'parameter_defaults': {'value1': {'foo': 1}}, 'parameter_merge_strategies': {'value1': 'deep_merge'}}
    env2 = {'parameter_defaults': {'value1': {'foo': 0}}}
    files = {'env_1': json.dumps(env1), 'env_2': json.dumps(env2)}
    environment_files = ['env_1', 'env_2']
    param_schemata = {'value1': parameters.Schema(parameters.Schema.MAP)}
    env_util.merge_environments(environment_files, files, self.params, param_schemata)
    self.assertEqual({'value1': {'foo': 0}}, self.params['parameter_defaults'])