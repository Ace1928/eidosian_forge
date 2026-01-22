import unittest
def test_pickle_maxsize(self):
    import pickle
    import sys
    for n in [0, 1, sys.getrecursionlimit() * 2]:
        source = self.Cache(maxsize=n)
        source.update(((i, i) for i in range(n)))
        cache = pickle.loads(pickle.dumps(source))
        self.assertEqual(n, len(cache))
        self.assertEqual(source, cache)