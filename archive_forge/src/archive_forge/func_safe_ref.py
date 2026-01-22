import operator
import sys
import traceback
import weakref
def safe_ref(target, on_delete=None):
    """Return a *safe* weak reference to a callable target.

    - ``target``: The object to be weakly referenced, if it's a bound
      method reference, will create a BoundMethodWeakref, otherwise
      creates a simple weakref.

    - ``on_delete``: If provided, will have a hard reference stored to
      the callable to be called after the safe reference goes out of
      scope with the reference object, (either a weakref or a
      BoundMethodWeakref) as argument.
    """
    try:
        im_self = get_self(target)
    except AttributeError:
        if callable(on_delete):
            return weakref.ref(target, on_delete)
        else:
            return weakref.ref(target)
    else:
        if im_self is not None:
            assert hasattr(target, 'im_func') or hasattr(target, '__func__'), f"safe_ref target {target!r} has im_self, but no im_func, don't know how to create reference"
            reference = BoundMethodWeakref(target=target, on_delete=on_delete)
            return reference