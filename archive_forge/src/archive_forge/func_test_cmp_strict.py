import unittest
from distutils.version import LooseVersion
from distutils.version import StrictVersion
def test_cmp_strict(self):
    versions = (('1.5.1', '1.5.2b2', -1), ('161', '3.10a', ValueError), ('8.02', '8.02', 0), ('3.4j', '1996.07.12', ValueError), ('3.2.pl0', '3.1.1.6', ValueError), ('2g6', '11g', ValueError), ('0.9', '2.2', -1), ('1.2.1', '1.2', 1), ('1.1', '1.2.2', -1), ('1.2', '1.1', 1), ('1.2.1', '1.2.2', -1), ('1.2.2', '1.2', 1), ('1.2', '1.2.2', -1), ('0.4.0', '0.4', 0), ('1.13++', '5.5.kw', ValueError))
    for v1, v2, wanted in versions:
        try:
            res = StrictVersion(v1)._cmp(StrictVersion(v2))
        except ValueError:
            if wanted is ValueError:
                continue
            else:
                raise AssertionError("cmp(%s, %s) shouldn't raise ValueError" % (v1, v2))
        self.assertEqual(res, wanted, 'cmp(%s, %s) should be %s, got %s' % (v1, v2, wanted, res))
        res = StrictVersion(v1)._cmp(v2)
        self.assertEqual(res, wanted, 'cmp(%s, %s) should be %s, got %s' % (v1, v2, wanted, res))
        res = StrictVersion(v1)._cmp(object())
        self.assertIs(res, NotImplemented, 'cmp(%s, %s) should be NotImplemented, got %s' % (v1, v2, res))