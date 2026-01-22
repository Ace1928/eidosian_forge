from eventlet.event import Event
from eventlet import greenthread
import collections
def spawn_many(self, depends, function, *args, **kwds):
    """
        spawn_many() accepts a single *function* whose parameters are the same
        as for :meth:`spawn`.

        The difference is that spawn_many() accepts a dependency dict
        *depends*. A new greenthread is spawned for each key in the dict. That
        dict key's value should be an iterable of other keys on which this
        greenthread depends.

        If the *depends* dict contains any key already passed to :meth:`spawn`
        or :meth:`post`, spawn_many() raises :class:`Collision`. It is
        indeterminate how many of the other keys in *depends* will have
        successfully spawned greenthreads.
        """
    for key, deps in depends.items():
        self.spawn(key, deps, function, *args, **kwds)