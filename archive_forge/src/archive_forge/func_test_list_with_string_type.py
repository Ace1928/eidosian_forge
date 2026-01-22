from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_list_with_string_type():
    field = List('graphene.types.tests.utils.MyLazyType')
    assert field.of_type == MyLazyType