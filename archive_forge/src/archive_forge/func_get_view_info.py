import re
from . import errors, osutils, transport
def get_view_info(self):
    """Get the current view and dictionary of views.

        Returns: current, views where
          current = the name of the current view or None if no view is enabled
          views = a map from view name to list of files/directories
        """
    self._load_view_info()
    return (self._current, self._views)