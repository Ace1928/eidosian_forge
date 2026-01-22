import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_format_labels_multiple_colon_values(self):
    la = utils.format_labels(['K1=V1', 'K2=V2,V22,V222,V2222', 'K3=3.3.3.3'])
    self.assertEqual({'K1': 'V1', 'K2': 'V2,V22,V222,V2222', 'K3': '3.3.3.3'}, la)