import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@classmethod
def create_profile_dir_by_name(cls, path, name=u'default', config=None):
    """Create a profile dir by profile name and path.

        Parameters
        ----------
        path : unicode
            The path (directory) to put the profile directory in.
        name : unicode
            The name of the profile.  The name of the profile directory will
            be "profile_<profile>".
        """
    if not os.path.isdir(path):
        raise ProfileDirError('Directory not found: %s' % path)
    profile_dir = os.path.join(path, u'profile_' + name)
    return cls(location=profile_dir, config=config)