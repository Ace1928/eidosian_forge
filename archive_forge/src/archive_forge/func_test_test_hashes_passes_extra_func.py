import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_hashes_passes_extra_func(self):
    mock_extra_func = mock.Mock()
    with mock.patch.object(self.ovc, 'get_hashes') as mock_get_hashes:
        self.ovc.test_hashes({}, extra_data_func=mock_extra_func)
        mock_get_hashes.assert_called_once_with(extra_data_func=mock_extra_func)