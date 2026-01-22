import collections.abc
import typing
def is_forward_ref(tp):
    """Returns true if `tp` is a typing forward reference."""
    if hasattr(typing, 'ForwardRef'):
        return isinstance(tp, typing.ForwardRef)
    elif hasattr(typing, '_ForwardRef'):
        return isinstance(tp, typing._ForwardRef)
    else:
        return False