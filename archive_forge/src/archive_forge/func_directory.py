import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
@staticmethod
def directory(path):
    """Loader factory for loading templates from a local directory.
        
        :param path: the path to the local directory containing the templates
        :return: the loader function to load templates from the given directory
        :rtype: ``function``
        """

    def _load_from_directory(filename):
        filepath = os.path.join(path, filename)
        fileobj = open(filepath, 'rb')
        mtime = os.path.getmtime(filepath)

        def _uptodate():
            return mtime == os.path.getmtime(filepath)
        return (filepath, filename, fileobj, _uptodate)
    return _load_from_directory