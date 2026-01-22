import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_no_mutable_default_args(self):
    self.assertEqual(1, len(list(checks.no_mutable_default_args(' def fake_suds_context(calls={}):'))))
    self.assertEqual(1, len(list(checks.no_mutable_default_args('def get_info_from_bdm(virt_type, bdm, mapping=[])'))))
    self.assertEqual(0, len(list(checks.no_mutable_default_args('defined = []'))))
    self.assertEqual(0, len(list(checks.no_mutable_default_args('defined, undefined = [], {}'))))