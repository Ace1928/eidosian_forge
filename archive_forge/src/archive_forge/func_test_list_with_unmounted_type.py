from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_list_with_unmounted_type():
    with raises(Exception) as exc_info:
        List(String())
    assert str(exc_info.value) == 'List could not have a mounted String() as inner type. Try with List(String).'