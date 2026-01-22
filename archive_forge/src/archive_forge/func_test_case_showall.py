import unittest
from IPython.utils import wildcard
def test_case_showall(self):
    ns = root.__dict__
    tests = [('a*', ['abbot', 'abel', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop']), ('_a*', ['_apan']), ('_*anka', ['__anka']), ('_*a*', ['__anka', '_apan'])]
    for pat, res in tests:
        res.sort()
        a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=False, show_all=True).keys())
        self.assertEqual(a, res)