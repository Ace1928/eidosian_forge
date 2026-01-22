import _imp
import _io
import sys
import _warnings
import marshal
def find_loader(self, fullname):
    """Try to find a loader for the specified module, or the namespace
        package portions. Returns (loader, list-of-portions).

        This method is deprecated.  Use find_spec() instead.

        """
    _warnings.warn('FileFinder.find_loader() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
    spec = self.find_spec(fullname)
    if spec is None:
        return (None, [])
    return (spec.loader, spec.submodule_search_locations or [])