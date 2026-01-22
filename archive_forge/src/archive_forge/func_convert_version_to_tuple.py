import functools
import re
import packaging.version
from oslo_utils._i18n import _
def convert_version_to_tuple(version_str):
    """Convert a version string with dots to a tuple.

    .. versionadded:: 2.0
    """
    version_str = re.sub('(\\d+)(a|alpha|b|beta|rc)\\d+$', '\\1', version_str)
    return tuple((int(part) for part in version_str.split('.')))