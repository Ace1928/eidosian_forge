import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_args_array_to_patch_format_error(self):
    my_args = {'attributes': ['foobar'], 'op': 'add'}
    self.assertRaises(exc.CommandError, utils.args_array_to_patch, my_args['op'], my_args['attributes'])