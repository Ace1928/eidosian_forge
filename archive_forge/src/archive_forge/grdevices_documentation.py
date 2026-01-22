import io
import os
import tempfile
from contextlib import contextmanager
from rpy2.robjects.packages import importr, WeakPackage

    Context manager that returns a R figures in a :class:`io.BytesIO`
    object.

    :param device: an R "device" function. This function is expected
                   to take a filename as its first argument.

    