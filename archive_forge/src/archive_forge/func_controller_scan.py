import os
import re
import six
from six.moves import urllib
from routes import request_config
def controller_scan(directory=None):
    """Scan a directory for python files and use them as controllers"""
    if directory is None:
        return []

    def find_controllers(dirname, prefix=''):
        """Locate controllers in a directory"""
        controllers = []
        for fname in os.listdir(dirname):
            filename = os.path.join(dirname, fname)
            if os.path.isfile(filename) and re.match('^[^_]{1,1}.*\\.py$', fname):
                controllers.append(prefix + fname[:-3])
            elif os.path.isdir(filename):
                controllers.extend(find_controllers(filename, prefix=prefix + fname + '/'))
        return controllers
    controllers = find_controllers(directory)
    controllers.sort(key=len, reverse=True)
    return controllers