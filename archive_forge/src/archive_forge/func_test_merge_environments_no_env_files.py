import json
from heat.common import environment_util as env_util
from heat.common import exception
from heat.engine import parameters
from heat.tests import common
from heat.tests import utils
def test_merge_environments_no_env_files(self):
    files = {'env_1': json.dumps(self.env_1)}
    env_util.merge_environments(None, files, self.params, self.param_schemata)
    expected = {'parameters': {}, 'resource_registry': {}, 'parameter_defaults': {}}
    self.assertEqual(expected, self.params)