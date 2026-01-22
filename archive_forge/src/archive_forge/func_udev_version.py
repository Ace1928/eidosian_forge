import errno
import os
import stat
import sys
from subprocess import check_output
def udev_version():
    """
    Get the version of the underlying udev library.

    udev doesn't use a standard major-minor versioning scheme, but instead
    labels releases with a single consecutive number.  Consequently, the
    version number returned by this function is a single integer, and not a
    tuple (like for instance the interpreter version in
    :data:`sys.version_info`).

    As libudev itself does not provide a function to query the version number,
    this function calls the ``udevadm`` utility, so be prepared to catch
    :exc:`~exceptions.EnvironmentError` and
    :exc:`~subprocess.CalledProcessError` if you call this function.

    Return the version number as single integer.  Raise
    :exc:`~exceptions.ValueError`, if the version number retrieved from udev
    could not be converted to an integer.  Raise
    :exc:`~exceptions.EnvironmentError`, if ``udevadm`` was not found, or could
    not be executed.  Raise :exc:`subprocess.CalledProcessError`, if
    ``udevadm`` returned a non-zero exit code.  On Python 2.7 or newer, the
    ``output`` attribute of this exception is correctly set.

    .. versionadded:: 0.8
    """
    output = ensure_unicode_string(check_output(['udevadm', '--version']))
    return int(output.strip())