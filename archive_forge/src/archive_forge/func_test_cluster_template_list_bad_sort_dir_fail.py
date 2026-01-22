import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_list_bad_sort_dir_fail(self):
    arglist = ['--sort-dir', 'foo']
    verifylist = [('limit', None), ('sort_key', None), ('sort_dir', 'foo'), ('fields', None)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)