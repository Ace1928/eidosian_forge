import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def conditional_dispatcher(default_func: Optional[Callable[..., Any]]=None, entry_point: Optional[str]=None) -> Callable:
    """Decorating a conditional dispatcher that will run the **first matching** registered
    functions in other modules/packages. This is a more general solution compared to
    ``functools.singledispatch``. You can write arbitrary matching functions according
    to all the inputs of the function.

    .. admonition:: Examples

        Assume in ``pkg1.module1``, you have:

        .. code-block:: python

            from triad import conditional_dispatcher

            @conditional_dispatcher(entry_point="my.plugins")
            def get_len(obj):
                raise NotImplementedError

        In another package ``pkg2``, in ``setup.py``, you define
        an entry point as:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And in ``pkg2.module2``:

        .. code-block:: python

            from pkg1.module1 import get_len

            @get_len.candidate(lambda obj: isinstance(obj, str))
            def get_str_len(obj:str) -> int:
                return len(obj)

            @get_len.candidate(lambda obj: isinstance(obj, int) and obj == 10)
            def get_int_len(obj:int) -> int:
                return obj

        Now, both functions will be automatically registered when ``pkg2``
        is installed in the environement. In another ``pkg3``:

        .. code-block:: python

            from pkg1.module1 import get_len

            assert get_len("abc") == 3  # calling get_str_len
            assert get_len(10) == 10  # calling get_int_len
            get_len(20)  # raise NotImplementedError due to no matching candidates

    .. seealso::

        Please read :meth:`~.ConditionalDispatcher.candidate` for details about the
        matching function and priority settings.

    :param default_func: the function to decorate
    :param entry_point: the entry point to preload dispatchers, defaults to None
    """
    return (lambda func: ConditionalDispatcher(func, is_broadcast=False, entry_point=entry_point)) if default_func is None else ConditionalDispatcher(default_func, is_broadcast=False, entry_point=entry_point)