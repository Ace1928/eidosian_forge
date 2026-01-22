import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
def test_load_balancer_set_tag(self, mock_attrs):
    self.api_mock.load_balancer_show.return_value = {'tags': ['foo']}
    mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'tags': ['bar']}
    arglist = [self._lb.id, '--tag', 'bar']
    verifylist = [('loadbalancer', self._lb.id), ('tags', ['bar'])]
    try:
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
    except Exception as e:
        self.fail('%s raised unexpectedly' % e)
    self.api_mock.load_balancer_set.assert_called_once()
    kwargs = self.api_mock.load_balancer_set.mock_calls[0][2]
    tags = kwargs['json']['loadbalancer']['tags']
    self.assertEqual(2, len(tags))
    self.assertIn('foo', tags)
    self.assertIn('bar', tags)