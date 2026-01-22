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
def test_cluster_upgrade_pass(self):
    cluster_template_id = 'TEMPLATE_ID'
    arglist = ['foo', cluster_template_id]
    verifylist = [('cluster', 'foo'), ('cluster_template', cluster_template_id), ('max_batch_size', 1), ('nodegroup', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.upgrade.assert_called_with('UUID1', cluster_template_id, 1, None)