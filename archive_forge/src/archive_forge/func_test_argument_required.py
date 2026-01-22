from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_argument_required():
    arg = Argument(String, required=True)
    assert arg.type == NonNull(String)