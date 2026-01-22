import re
from . import errors, osutils, transport
def check_path_in_view(tree, relpath):
    """If a working tree has a view enabled, check the path is within it."""
    if tree.supports_views():
        view_files = tree.views.lookup_view()
        if view_files and (not osutils.is_inside_any(view_files, relpath)):
            raise FileOutsideView(relpath, view_files)