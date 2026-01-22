from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_to_arguments():
    args = {'arg_string': Argument(String), 'unmounted_arg': String(required=True)}
    my_args = to_arguments(args)
    assert my_args == {'arg_string': Argument(String), 'unmounted_arg': Argument(String, required=True)}