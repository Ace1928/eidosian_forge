import unittest
from distutils.version import LooseVersion
from distutils.version import StrictVersion
def test_cmp(self):
    versions = (('1.5.1', '1.5.2b2', -1), ('161', '3.10a', 1), ('8.02', '8.02', 0), ('3.4j', '1996.07.12', -1), ('3.2.pl0', '3.1.1.6', 1), ('2g6', '11g', -1), ('0.960923', '2.2beta29', -1), ('1.13++', '5.5.kw', -1))
    for v1, v2, wanted in versions:
        res = LooseVersion(v1)._cmp(LooseVersion(v2))
        self.assertEqual(res, wanted, 'cmp(%s, %s) should be %s, got %s' % (v1, v2, wanted, res))
        res = LooseVersion(v1)._cmp(v2)
        self.assertEqual(res, wanted, 'cmp(%s, %s) should be %s, got %s' % (v1, v2, wanted, res))
        res = LooseVersion(v1)._cmp(object())
        self.assertIs(res, NotImplemented, 'cmp(%s, %s) should be NotImplemented, got %s' % (v1, v2, res))