from os.path import join, dirname, exists, abspath
from kivy import kivy_data_dir
from kivy.cache import Cache
from kivy.utils import platform
from kivy.logger import Logger
import sys
import os
import kivy
def resource_remove_path(path):
    """Remove a search path.

    .. versionadded:: 1.0.8
    """
    if path not in resource_paths:
        return
    Logger.debug('Resource: remove <%s> from path list' % path)
    resource_paths.remove(path)