import io
import os
import tempfile
from contextlib import contextmanager
from rpy2.robjects.packages import importr, WeakPackage
@contextmanager
def render_to_bytesio(device, *device_args, **device_kwargs):
    """
    Context manager that returns a R figures in a :class:`io.BytesIO`
    object.

    :param device: an R "device" function. This function is expected
                   to take a filename as its first argument.

    """
    fn = tempfile.mktemp()
    b = io.BytesIO()
    current = dev_cur()[0]
    try:
        device(fn, *device_args, **device_kwargs)
        yield b
    finally:
        if current != dev_cur()[0]:
            dev_off()
        if os.path.exists(fn):
            with open(fn, 'rb') as fh:
                b.write(fh.read())
            os.unlink(fn)