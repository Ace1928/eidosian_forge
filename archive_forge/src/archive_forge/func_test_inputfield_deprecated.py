from functools import partial
from pytest import raises
from ..inputfield import InputField
from ..structures import NonNull
from .utils import MyLazyType
def test_inputfield_deprecated():
    MyType = object()
    deprecation_reason = 'deprecated'
    field = InputField(MyType, required=False, deprecation_reason=deprecation_reason)
    assert isinstance(field.type, type(MyType))
    assert field.deprecation_reason == deprecation_reason