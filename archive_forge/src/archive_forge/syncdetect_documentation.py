import contextlib
import threading
import warnings
Allows or disallows device synchronization temporarily in the current thread.

    .. warning::

       This API has been deprecated in CuPy v10 and will be removed in future
       releases.

    If device synchronization is detected, :class:`cupyx.DeviceSynchronized`
    will be raised.

    Note that there can be false negatives and positives.
    Device synchronization outside CuPy will not be detected.
    