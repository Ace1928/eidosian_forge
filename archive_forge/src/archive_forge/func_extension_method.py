from typing import Callable, Dict, Optional, Type, get_type_hints, Any
import inspect
from triad.utils.assertion import assert_or_throw
def extension_method(func: Optional[Callable]=None, class_type: Optional[Type]=None, name: Optional[str]=None, on_dup: str='error') -> Callable:
    """The decorator to add functions as members of the
    correspondent classes.

    :param func: the function under the decorator
    :param class_type: the parent class type, defaults to None
    :param name: the specified class method name, defaults to None. If None
      then ``func.__name__`` will be used as the method name
    :param on_dup: action on name duplication, defaults to ``error``. ``error``
      will throw a ValueError; ``ignore`` will take no action; ``overwrite``
      will use the current method to overwrite.
    :return: the underlying function

    .. admonition:: Examples

        .. code-block:: python

            @extensible_class
            class A:

                # It's recommended to implement __getattr__ so that
                # PyLint will not complain about the dynamically added methods
                def __getattr__(self, name):
                    raise NotImplementedError

            # The simplest way to use this decorator, the first argument of
            # the method must be annotated, and the annotated type is the
            # class type to add this method to.
            @extension_method
            def method1(obj:A):
                return 1

            assert 1 == A().method1()

            # Or you can be explicit of the class type and the name of the
            # method in the class. In this case, you don't have to annotate
            # the first argument.
            @extension_method(class_type=A, name="m3")
            def method2(obj, b):
                return 2 + b

            assert 5 == A().m3(3)

    .. note::

        If the method name is already in the original class, a ValueError will be
        thrown. You can't modify any built-in attribute.
    """
    if func is not None:
        _CLASS_EXTENSIONS.add_method(_get_first_arg_type(func) if class_type is None else class_type, func=func, name=name, on_dup=on_dup)
        return func
    else:

        def inner(func):
            _CLASS_EXTENSIONS.add_method(_get_first_arg_type(func) if class_type is None else class_type, func=func, name=name, on_dup=on_dup)
            return func
        return inner