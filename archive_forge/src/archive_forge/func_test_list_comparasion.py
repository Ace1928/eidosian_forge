from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_list_comparasion():
    list1 = List(String)
    list2 = List(String)
    list3 = List(None)
    list1_argskwargs = List(String, None, b=True)
    list2_argskwargs = List(String, None, b=True)
    assert list1 == list2
    assert list1 != list3
    assert list1_argskwargs == list2_argskwargs
    assert list1 != list1_argskwargs