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
def test_cluster_resize_to_zero_pass(self):
    arglist = ['foo', '0']
    verifylist = [('cluster', 'foo'), ('node_count', 0), ('nodes_to_remove', None), ('nodegroup', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.resize.assert_called_with('UUID1', 0, None, None)