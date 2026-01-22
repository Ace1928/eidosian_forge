from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_argument_with_lazy_partial_type():
    MyType = object()
    arg = Argument(partial(lambda: MyType))
    assert arg.type == MyType