import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_handle_json_from_file_bad_json(self):
    contents = 'foo'
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(contents)
        f.flush()
        self.assertRaisesRegex(exc.InvalidAttribute, 'For JSON', utils.handle_json_from_file, f.name)