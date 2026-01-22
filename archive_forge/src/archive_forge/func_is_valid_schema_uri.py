import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
def is_valid_schema_uri(uri):
    """Reports if uri is suitable for use as a namespace schema URI.

    It must be non-empty and it must not contain white space.

    :param uri string
    :return bool
    """
    if len(uri) <= 0:
        return False
    return uri.find(' ') == -1