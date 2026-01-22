from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_params_extract_garbage_prefix(self):
    p = {'prefixParameters.member.Foo.Bar.ParameterKey': 'foo', 'Parameters.member.Foo.Bar.ParameterValue': 'bar'}
    params = api_utils.extract_param_pairs(p, prefix='Parameters', keyname='ParameterKey', valuename='ParameterValue')
    self.assertFalse(params)