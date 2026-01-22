import base64
import calendar
import datetime
import sys
import six
from six.moves import urllib
from google.auth import exceptions
def update_query(url, params, remove=None):
    """Updates a URL's query parameters.

    Replaces any current values if they are already present in the URL.

    Args:
        url (str): The URL to update.
        params (Mapping[str, str]): A mapping of query parameter
            keys to values.
        remove (Sequence[str]): Parameters to remove from the query string.

    Returns:
        str: The URL with updated query parameters.

    Examples:

        >>> url = 'http://example.com?a=1'
        >>> update_query(url, {'a': '2'})
        http://example.com?a=2
        >>> update_query(url, {'b': '3'})
        http://example.com?a=1&b=3
        >> update_query(url, {'b': '3'}, remove=['a'])
        http://example.com?b=3

    """
    if remove is None:
        remove = []
    parts = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parts.query)
    query_params.update(params)
    query_params = {key: value for key, value in six.iteritems(query_params) if key not in remove}
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)