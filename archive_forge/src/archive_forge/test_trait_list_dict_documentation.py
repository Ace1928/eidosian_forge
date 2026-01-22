import copy
import pickle
import unittest
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Dict, List, Set, Str, Int, Instance
 Test the persistence behavior of TraitListObjects, TraitDictObjects and
TraitSetObjects.
