import functools
import logging
from collections import abc
from typing import Union, Mapping, Any, Callable
from ray.rllib.core.models.specs.specs_base import Spec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.nested_dict import NestedDict
from ray.util.annotations import DeveloperAPI
@DeveloperAPI(stability='alpha')
def check_output_specs(output_specs: str, *, cache: bool=False):
    """A general-purpose spec checker decorator for Neural Network base classes.

    This is a stateful decorator
    (https://realpython.com/primer-on-python-decorators/#stateful-decorators) to
    enforce output specs for any instance method that outputs a single dict-like object.

    It also allows you to cache the validation to make sure the spec is only validated
    once in the entire lifetime of the instance.

    Examples (See more examples in ../tests/test_specs_dict.py):

    .. testcode::

        import torch
        from torch import nn
        from ray.rllib.core.models.specs.specs_base import TensorSpec

        class MyModel(nn.Module):
            @property
            def output_specs(self):
                return {"obs": TensorSpec("b, d", d=64)}

            @check_output_specs("output_specs")
            def forward(self, input_data, return_loss=False):
                return {"obs": torch.randn(32, 64)}

    Args:
        func: The instance method to decorate. It should be a callable that takes
            `self` as the first argument, `input_data` as the second argument and any
            other keyword argument thereafter. It should return a single dict-like
            object (i.e. not a tuple).
        output_specs: `self` should have an instance attribute whose name matches the
            string in output_specs and returns the `SpecDict`, `Spec`, or simply the
            `Type` that the `input_data` should comply with. It can alos be None or
            empty list / dict to enforce no input spec.
        cache: If True, only checks the data validation for the first time the
            instance method is called.

    Returns:
        A wrapped instance method. In case of `cache=True`, after the first invokation
        of the decorated method, the intance will have `__checked_output_specs_cache__`
        attribute that stores which method has been invoked at least once. This is a
        special attribute that can be used for the cache itself. The wrapped class
        method also has a special attribute `__checked_output_specs__` that marks the
        method as decorated.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, input_data, **kwargs):
            if cache and (not hasattr(self, '__checked_output_specs_cache__')):
                self.__checked_output_specs_cache__ = {}
            output_data = func(self, input_data, **kwargs)
            if output_specs:
                if hasattr(self, output_specs):
                    spec = getattr(self, output_specs)
                else:
                    raise ValueError(f'object {self} has no attribute {output_specs}.')
                if spec is not None:
                    spec = convert_to_canonical_format(spec)
                    _validate(cls_instance=self, method=func, data=output_data, spec=spec, tag='output')
            if cache and func.__name__ not in self.__checked_output_specs_cache__:
                self.__checked_output_specs_cache__[func.__name__] = True
            return output_data
        wrapper.__checked_output_specs__ = True
        return wrapper
    return decorator