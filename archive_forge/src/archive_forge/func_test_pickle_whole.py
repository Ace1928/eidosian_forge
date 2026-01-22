import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
def test_pickle_whole(self):
    a = A()
    pickle.loads(pickle.dumps(a))
    b = B(dict=dict(a=a))
    pickle.loads(pickle.dumps(b))