from collections.abc import MutableMapping
from weakref import ref
 A weak-key dictionary that uses the id() of the key for comparisons.

    This differs from `WeakIDDict` in that it does not try to make a weakref to
    the values.
    