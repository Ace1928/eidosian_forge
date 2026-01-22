import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def list_bundled_profiles():
    """list profiles that are bundled with IPython."""
    path = os.path.join(get_ipython_package_dir(), u'core', u'profile')
    profiles = []
    files = os.scandir(path)
    for profile in files:
        if profile.is_dir() and profile.name != '__pycache__':
            profiles.append(profile.name)
    return profiles