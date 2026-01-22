import copy
import os
import sys
import tempfile
from unittest import mock
from contextlib import contextmanager
from unittest.mock import call
from magnumclient import exceptions
from magnumclient.osc.v1 import clusters as osc_clusters
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_cluster_delete_bad_uuid(self):
    arglist = ['foo']
    verifylist = [('cluster', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    returns = self.cmd.take_action(parsed_args)
    self.assertEqual(returns, None)