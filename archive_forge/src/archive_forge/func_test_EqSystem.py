import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
@requires('numpy')
def test_EqSystem():
    a, b = sbstncs = (Substance('a'), Substance('b'))
    rxns = [Reaction({'a': 1}, {'b': 1})]
    es = EqSystem(rxns, collections.OrderedDict([(s.name, s) for s in sbstncs]))
    assert es.net_stoichs().tolist() == [[-1, 1]]