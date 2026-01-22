from .._cloudpickle_wrapper import wrap_non_picklable_objects
from .._cloudpickle_wrapper import _my_wrap_non_picklable_objects
def test_wrap_non_picklable_objects():
    for obj in (a_function, AClass()):
        wrapped_obj = wrap_non_picklable_objects(obj)
        my_wrapped_obj = _my_wrap_non_picklable_objects(obj)
        assert wrapped_obj(1) == my_wrapped_obj(1)