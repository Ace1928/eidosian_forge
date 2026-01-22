import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def recover_args(flattened_args):
    """Recreates `args` and `kwargs` from the flattened arg list.

    Args:
        flattened_args: List of args and kwargs. This should be the output of
            `flatten_args`.

    Returns:
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.
    """
    assert len(flattened_args) % 2 == 0, 'Flattened arguments need to be even-numbered. See `flatten_args`.'
    args = []
    kwargs = {}
    for name_index in range(0, len(flattened_args), 2):
        name, arg = (flattened_args[name_index], flattened_args[name_index + 1])
        if name == DUMMY_TYPE:
            args.append(arg)
        else:
            kwargs[name] = arg
    return (args, kwargs)