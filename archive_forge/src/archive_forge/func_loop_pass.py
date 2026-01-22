from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
def loop_pass(base_pass: Callable, n_iter: Optional[int]=None, predicate: Optional[Callable]=None):
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    assert (n_iter is not None) ^ (predicate is not None), 'Exactly one of `n_iter`or `predicate` must be specified.'

    @wraps(base_pass)
    def new_pass(source):
        output = source
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        else:
            raise RuntimeError(f'loop_pass must be given positive int n_iter (given {n_iter}) xor predicate (given {predicate})')
        return output
    return new_pass