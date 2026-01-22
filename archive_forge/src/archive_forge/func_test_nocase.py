import unittest
from IPython.utils import wildcard
def test_nocase(self):
    ns = root.__dict__
    tests = [('a*', ['abbot', 'abel', 'ABEL', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop', 'ABEL.koppel', 'ABEL.loop']), ('_a*', []), ('_*anka', ['__anka', '__ANKA']), ('_*a*', ['__anka', '__ANKA'])]
    for pat, res in tests:
        res.sort()
        a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=True, show_all=False).keys())
        self.assertEqual(a, res)