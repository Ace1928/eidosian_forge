import os
import mimetypes
from datetime import datetime
from time import gmtime
def get_directory_loader(self, directory):

    def loader(path):
        path = path or directory
        if path is not None:
            path = os.path.join(directory, path)
        if os.path.isfile(path):
            return (os.path.basename(path), self._opener(path))
        return (None, None)
    return loader