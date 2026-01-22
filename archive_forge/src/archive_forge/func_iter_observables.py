import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def iter_observables(self, object):
    if object is None:
        raise ValueError('This observer cannot handle None.')
    yield from ()