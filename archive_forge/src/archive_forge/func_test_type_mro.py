from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
def test_type_mro():
    assert super_signature([[object], [type]]) == [type]