import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def expand_sig(sig):
    """ Convert the signature spec in ``module_info`` to add to ``signatures``

    The input signature spec is one of:
        - ``lambda_func``
        - ``(num_position_args, lambda_func)``
        - ``(num_position_args, lambda_func, keyword_only_args)``

    The output signature spec is:
        ``(num_position_args, lambda_func, keyword_exclude, sigspec)``

    where ``keyword_exclude`` includes keyword only arguments and, if variadic
    keywords is present, the names of position-only argument.  The latter is
    included to support builtins such as ``partial(func, *args, **kwargs)``,
    which allows ``func=`` to be used as a keyword even though it's the name
    of a positional argument.
    """
    if isinstance(sig, tuple):
        if len(sig) == 3:
            num_pos_only, func, keyword_only = sig
            assert isinstance(sig[-1], tuple)
        else:
            num_pos_only, func = sig
            keyword_only = ()
        sigspec = signature_or_spec(func)
    else:
        func = sig
        sigspec = signature_or_spec(func)
        num_pos_only = num_pos_args(sigspec)
        keyword_only = ()
    keyword_exclude = get_exclude_keywords(num_pos_only, sigspec)
    return (num_pos_only, func, keyword_only + keyword_exclude, sigspec)