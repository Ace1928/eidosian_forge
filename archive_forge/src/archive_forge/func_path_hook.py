import _imp
import _io
import sys
import _warnings
import marshal
@classmethod
def path_hook(cls, *loader_details):
    """A class method which returns a closure to use on sys.path_hook
        which will return an instance using the specified loaders and the path
        called on the closure.

        If the path called on the closure is not a directory, ImportError is
        raised.

        """

    def path_hook_for_FileFinder(path):
        """Path hook for importlib.machinery.FileFinder."""
        if not _path_isdir(path):
            raise ImportError('only directories are supported', path=path)
        return cls(path, *loader_details)
    return path_hook_for_FileFinder