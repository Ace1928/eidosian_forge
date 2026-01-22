import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_delete_container_single(self):
    self.requests_mock.register_uri('DELETE', object_fakes.ENDPOINT + '/ernie', status_code=200)
    arglist = ['ernie']
    verifylist = [('containers', ['ernie'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ret = self.cmd.take_action(parsed_args)
    self.assertIsNone(ret)