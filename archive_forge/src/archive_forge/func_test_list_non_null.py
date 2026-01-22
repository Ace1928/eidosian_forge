from functools import partial
from ..dynamic import Dynamic
from ..scalars import String
from ..structures import List, NonNull
def test_list_non_null():
    dynamic = Dynamic(lambda: List(NonNull(String)))
    assert dynamic.get_type().of_type.of_type == String
    assert str(dynamic.get_type()) == '[String!]'