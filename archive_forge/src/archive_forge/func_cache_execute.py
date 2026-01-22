import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging
from cachetools import LRUCache, Cache
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from .set_shots import set_shots
from .jacobian_products import (
def cache_execute(fn: Callable, cache, pass_kwargs=False, return_tuple=True, expand_fn=None):
    """Decorator that adds caching to a function that executes
    multiple tapes on a device.

    This decorator makes use of :attr:`.QuantumTape.hash` to identify
    unique tapes.

    - If a tape does not match a hash in the cache, then the tape
      has not been previously executed. It is executed, and the result
      added to the cache.

    - If a tape matches a hash in the cache, then the tape has been previously
      executed. The corresponding cached result is
      extracted, and the tape is not passed to the execution function.

    - Finally, there might be the case where one or more tapes in the current
      set of tapes to be executed are identical and thus share a hash. If this is the case,
      duplicates are removed, to avoid redundant evaluations.

    Args:
        fn (callable): The execution function to add caching to.
            This function should have the signature ``fn(tapes, **kwargs)``,
            and it should return ``list[tensor_like]``, with the
            same length as the input ``tapes``.
        cache (None or dict or Cache or bool): The cache to use. If ``None``,
            caching will not occur.
        pass_kwargs (bool): If ``True``, keyword arguments passed to the
            wrapped function will be passed directly to ``fn``. If ``False``,
            they will be ignored.
        return_tuple (bool): If ``True``, the output of ``fn`` is returned
            as a tuple ``(fn_ouput, [])``, to match the output of execution functions
            that also return gradients.

    Returns:
        function: a wrapped version of the execution function ``fn`` with caching
        support
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Entry with args=(fn=%s, cache=%s, pass_kwargs=%s, return_tuple=%s, expand_fn=%s) called by=%s', fn if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(fn)) else '\n' + inspect.getsource(fn), cache, pass_kwargs, return_tuple, expand_fn if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(expand_fn)) else '\n' + inspect.getsource(expand_fn) + '\n', '::L'.join((str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3])))
    if expand_fn is not None:
        original_fn = fn

        def fn(tapes: Sequence[QuantumTape], **kwargs):
            tapes = [expand_fn(tape) for tape in tapes]
            return original_fn(tapes, **kwargs)

    @wraps(fn)
    def wrapper(tapes: Sequence[QuantumTape], **kwargs):
        if not pass_kwargs:
            kwargs = {}
        if cache is None or (isinstance(cache, bool) and (not cache)):
            res = list(fn(tapes, **kwargs))
            return (res, []) if return_tuple else res
        execution_tapes = {}
        cached_results = {}
        hashes = {}
        repeated = {}
        for i, tape in enumerate(tapes):
            h = tape.hash
            if h in hashes.values():
                idx = list(hashes.keys())[list(hashes.values()).index(h)]
                repeated[i] = idx
                continue
            hashes[i] = h
            if hashes[i] in cache:
                cached_results[i] = cache[hashes[i]]
                if tape.shots and getattr(cache, '_persistent_cache', True):
                    warnings.warn("Cached execution with finite shots detected!\nNote that samples as well as all noisy quantities computed via sampling will be identical across executions. This situation arises where tapes are executed with identical operations, measurements, and parameters.\nTo avoid this behavior, provide 'cache=False' to the QNode or execution function.", UserWarning)
            else:
                execution_tapes[i] = tape
        if not execution_tapes:
            if not repeated:
                res = list(cached_results.values())
                return (res, []) if return_tuple else res
        else:
            res = list(fn(tuple(execution_tapes.values()), **kwargs))
        final_res = []
        for i, tape in enumerate(tapes):
            if i in cached_results:
                final_res.append(cached_results[i])
            elif i in repeated:
                final_res.append(final_res[repeated[i]])
            else:
                r = res.pop(0)
                final_res.append(r)
                cache[hashes[i]] = r
        return (final_res, []) if return_tuple else final_res
    wrapper.fn = fn
    return wrapper