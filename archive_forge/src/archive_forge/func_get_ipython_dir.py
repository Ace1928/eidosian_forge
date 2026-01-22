import os.path
import tempfile
from warnings import warn
import IPython
from IPython.utils.importstring import import_item
from IPython.utils.path import (
def get_ipython_dir() -> str:
    """Get the IPython directory for this platform and user.

    This uses the logic in `get_home_dir` to find the home directory
    and then adds .ipython to the end of the path.
    """
    env = os.environ
    pjoin = os.path.join
    ipdir_def = '.ipython'
    home_dir = get_home_dir()
    xdg_dir = get_xdg_dir()
    if 'IPYTHON_DIR' in env:
        warn('The environment variable IPYTHON_DIR is deprecated since IPython 3.0. Please use IPYTHONDIR instead.', DeprecationWarning)
    ipdir = env.get('IPYTHONDIR', env.get('IPYTHON_DIR', None))
    if ipdir is None:
        ipdir = pjoin(home_dir, ipdir_def)
        if xdg_dir:
            xdg_ipdir = pjoin(xdg_dir, 'ipython')
            if _writable_dir(xdg_ipdir):
                cu = compress_user
                if os.path.exists(ipdir):
                    warn('Ignoring {0} in favour of {1}. Remove {0} to get rid of this message'.format(cu(xdg_ipdir), cu(ipdir)))
                elif os.path.islink(xdg_ipdir):
                    warn('{0} is deprecated. Move link to {1} to get rid of this message'.format(cu(xdg_ipdir), cu(ipdir)))
                else:
                    ipdir = xdg_ipdir
    ipdir = os.path.normpath(os.path.expanduser(ipdir))
    if os.path.exists(ipdir) and (not _writable_dir(ipdir)):
        warn("IPython dir '{0}' is not a writable location, using a temp directory.".format(ipdir))
        ipdir = tempfile.mkdtemp()
    elif not os.path.exists(ipdir):
        parent = os.path.dirname(ipdir)
        if not _writable_dir(parent):
            warn("IPython parent '{0}' is not a writable location, using a temp directory.".format(parent))
            ipdir = tempfile.mkdtemp()
        else:
            os.makedirs(ipdir, exist_ok=True)
    assert isinstance(ipdir, str), 'all path manipulation should be str(unicode), but are not.'
    return ipdir