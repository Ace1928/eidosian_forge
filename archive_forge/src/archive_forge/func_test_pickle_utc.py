import unittest
from six.moves import cPickle as pickle
import isodate
def test_pickle_utc(self):
    """
        isodate.UTC objects remain the same after pickling.
        """
    self.assertTrue(isodate.UTC is pickle.loads(pickle.dumps(isodate.UTC)))