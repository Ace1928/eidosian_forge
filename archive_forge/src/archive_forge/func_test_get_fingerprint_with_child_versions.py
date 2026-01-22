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
def test_get_fingerprint_with_child_versions(self):
    child_versions = {'1.0': '1.0', '1.1': '1.1'}
    MyObject.VERSION = '1.1'
    MyObject.child_versions = child_versions
    argspec = 'onix'
    with mock.patch.object(fixture, 'get_method_spec') as mock_gas:
        mock_gas.return_value = argspec
        fp = self.ovc._get_fingerprint(MyObject.__name__)
    exp_fields = sorted(list(MyObject.fields.items()))
    exp_methods = sorted([('remotable_method', argspec), ('remotable_classmethod', argspec)])
    exp_child_versions = collections.OrderedDict(sorted(child_versions.items()))
    exp_relevant_data = (exp_fields, exp_methods, exp_child_versions)
    expected_hash = hashlib.md5(bytes(repr(exp_relevant_data).encode())).hexdigest()
    expected_fp = '%s-%s' % (MyObject.VERSION, expected_hash)
    self.assertEqual(expected_fp, fp, '_get_fingerprint() did not generate a correct fingerprint.')