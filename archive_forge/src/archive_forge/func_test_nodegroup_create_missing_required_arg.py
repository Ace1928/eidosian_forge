import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_create_missing_required_arg(self):
    """Verifies missing required arguments."""
    arglist = [self.nodegroup.name]
    verifylist = [('name', self.nodegroup.name)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)