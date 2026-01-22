import re
from . import abnf_regexp
important_characters = {
def merge_paths(base_uri, relative_path):
    """Merge a base URI's path with a relative URI's path."""
    if base_uri.path is None and base_uri.authority is not None:
        return '/' + relative_path
    else:
        path = base_uri.path or ''
        index = path.rfind('/')
        return path[:index] + '/' + relative_path