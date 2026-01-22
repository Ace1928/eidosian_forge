import re
from . import errors, osutils, transport
def lookup_view(self, view_name=None):
    """Return the contents of a view.

        Args:
          view_Name: name of the view or None to lookup the current view

        Returns:
          the list of files/directories in the requested view
        """
    self._load_view_info()
    try:
        if view_name is None:
            if self._current:
                view_name = self._current
            else:
                return []
        return self._views[view_name]
    except KeyError:
        raise NoSuchView(view_name)