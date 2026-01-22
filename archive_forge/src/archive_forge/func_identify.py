import re
from . import lazy_regex
from .trace import mutter, warning
@staticmethod
def identify(pattern):
    """Returns pattern category.

        :param pattern: normalized pattern.
        Identify if a pattern is fullpath, basename or extension
        and returns the appropriate type.
        """
    if pattern.startswith('RE:') or '/' in pattern:
        return 'fullpath'
    elif pattern.startswith('*.'):
        return 'extension'
    else:
        return 'basename'