import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_format_labels_multiple_values_per_labels(self):
    la = utils.format_labels(['K1=V1', 'K1=V2'])
    self.assertEqual({'K1': 'V1,V2'}, la)