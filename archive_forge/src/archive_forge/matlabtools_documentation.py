import os
import re
import tempfile
import numpy as np
Returns a temporary file-like object with valid matlab name.

    The file name is accessible as the .name attribute of the returned object.
    The caller is responsible for closing the returned object, at which time
    the underlying file gets deleted from the filesystem.

    Parameters
    ----------

      dir : str
        A path to use as the starting directory.  Note that this directory must
        already exist, it is NOT created if it doesn't (in that case, OSError
        is raised instead).

    Returns
    -------
      f : A file-like object.

    Examples
    --------

    >>> fn = mlab_tempfile()
    >>> import os
    >>> filename = os.path.basename(fn.name)
    >>> '-' not in filename
    True
    >>> fn.close()

    