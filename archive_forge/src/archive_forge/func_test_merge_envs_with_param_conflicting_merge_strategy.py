import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_envs_with_param_conflicting_merge_strategy(self):
    merge_strategies = {'default': 'overwrite', 'lst_value1': 'merge', 'json_value1': 'deep_merge'}
    env3_merge_strategies = {'default': 'overwrite', 'lst_value1': 'deep_merge', 'json_value1': 'merge'}
    self.env_1['parameter_merge_strategies'] = merge_strategies
    self.env_3['parameter_merge_strategies'] = env3_merge_strategies
    files = {'env_1': json.dumps(self.env_1), 'env_2': json.dumps(self.env_2), 'env_3': json.dumps(self.env_3)}
    environment_files = ['env_1', 'env_2', 'env_3']
    self.assertRaises(exception.ConflictingMergeStrategyForParam, env_util.merge_environments, environment_files, files, self.params, self.param_schemata)