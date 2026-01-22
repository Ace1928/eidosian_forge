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
def check_input_specs(input_specs: str, *, only_check_on_retry: bool=True, filter: bool=False, cache: bool=False):
    """A general-purpose spec checker decorator for neural network base classes.

    This is a stateful decorator
    (https://realpython.com/primer-on-python-decorators/#stateful-decorators) to
    enforce input specs for any instance method that has an argument named
    `input_data` in its args.

    It also allows you to filter the input data dictionary to only include those keys
    that are specified in the model specs. It also allows you to cache the validation
    to make sure the spec is only validated once in the entire lifetime of the instance.

    See more examples in ../tests/test_specs_dict.py)

    .. testcode::

        import torch
        from torch import nn
        from ray.rllib.core.models.specs.specs_base import TensorSpec

        class MyModel(nn.Module):
            @property
            def input_specs(self):
                return {"obs": TensorSpec("b, d", d=64)}

            @check_input_specs("input_specs", only_check_on_retry=False)
            def forward(self, input_data, return_loss=False):
                ...

        model = MyModel()
        model.forward({"obs": torch.randn(32, 64)})

        # The following would raise an Error
        # model.forward({"obs": torch.randn(32, 32)})

    Args:
        func: The instance method to decorate. It should be a callable that takes
            `self` as the first argument, `input_data` as the second argument and any
            other keyword argument thereafter.
        input_specs: `self` should have an instance attribute whose name matches the
            string in input_specs and returns the `SpecDict`, `Spec`, or simply the
            `Type` that the `input_data` should comply with. It can also be None or
            empty list / dict to enforce no input spec.
        only_check_on_retry: If True, the spec will not be checked. Only if the
            decorated method raises an Exception, we check the spec to provide a more
            informative error message.
        filter: If True, and `input_data` is a nested dict the `input_data` will be
            filtered by its corresponding spec tree structure and then passed into the
            implemented function to make sure user is not confounded with unnecessary
            data.
        cache: If True, only checks the data validation for the first time the
            instance method is called.

    Returns:
        A wrapped instance method. In case of `cache=True`, after the first invokation
        of the decorated method, the intance will have `__checked_input_specs_cache__`
        attribute that stores which method has been invoked at least once. This is a
        special attribute that can be used for the cache itself. The wrapped class
        method also has a special attribute `__checked_input_specs__` that marks the
        method as decorated.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, input_data, **kwargs):
            if cache and (not hasattr(self, '__checked_input_specs_cache__')):
                self.__checked_input_specs_cache__ = {}
            if cache and func.__name__ not in self.__checked_input_specs_cache__:
                self.__checked_input_specs_cache__[func.__name__] = True
            initial_exception = None
            if only_check_on_retry:
                try:
                    return func(self, input_data, **kwargs)
                except SpecCheckingError as e:
                    raise e
                except Exception as e:
                    initial_exception = e
                    logger.error(f'Exception {e} raised on function call without checkin input specs. RLlib will now attempt to check the spec before calling the function again.')
            checked_data = input_data
            if input_specs:
                if hasattr(self, input_specs):
                    spec = getattr(self, input_specs)
                else:
                    raise SpecCheckingError(f'object {self} has no attribute {input_specs}.')
                if spec is not None:
                    spec = convert_to_canonical_format(spec)
                    checked_data = _validate(cls_instance=self, method=func, data=input_data, spec=spec, filter=filter, tag='input')
                    if filter and isinstance(checked_data, NestedDict):
                        checked_data = checked_data.filter(spec)
            if initial_exception:
                raise initial_exception
            return func(self, checked_data, **kwargs)
        wrapper.__checked_input_specs__ = True
        return wrapper
    return decorator