import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def head_from_fun(fun: Callable[..., Any], bound: bool=False) -> str:
    """Generate signature function from actual function."""
    is_function = inspect.isfunction(fun)
    is_callable = callable(fun)
    is_cython = fun.__class__.__name__ == 'cython_function_or_method'
    is_method = inspect.ismethod(fun)
    if not is_function and is_callable and (not is_method) and (not is_cython):
        name, fun = (fun.__class__.__name__, fun.__call__)
    else:
        name = fun.__name__
    definition = FUNHEAD_TEMPLATE.format(fun_name=name, fun_args=_argsfromspec(inspect.getfullargspec(fun)), fun_value=1)
    logger.debug(definition)
    namespace = {'__name__': fun.__module__}
    exec(definition, namespace)
    result = namespace[name]
    result._source = definition
    if bound:
        return partial(result, object())
    return result