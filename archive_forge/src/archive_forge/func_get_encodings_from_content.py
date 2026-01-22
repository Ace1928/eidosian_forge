import re
import sys
from requests import utils
def get_encodings_from_content(content):
    """Return encodings from given content string.

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import deprecated

        r = requests.get(url)
        encodings = deprecated.get_encodings_from_content(r)

    :param content: bytestring to extract encodings from
    :type content: bytes
    :return: encodings detected in the provided content
    :rtype: list(str)
    """
    encodings = find_charset(content) + find_pragma(content) + find_xml(content)
    if (3, 0) <= sys.version_info < (4, 0):
        encodings = [encoding.decode('utf8') for encoding in encodings]
    return encodings