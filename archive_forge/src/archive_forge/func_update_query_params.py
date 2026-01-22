import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def update_query_params(uri, params):
    """Updates a URI with new query parameters.

    If a given key from ``params`` is repeated in the ``uri``, then
    the URI will be considered invalid and an error will occur.

    If the URI is valid, then each value from ``params`` will
    replace the corresponding value in the query parameters (if
    it exists).

    Args:
        uri: string, A valid URI, with potential existing query parameters.
        params: dict, A dictionary of query parameters.

    Returns:
        The same URI but with the new query parameters added.
    """
    parts = urllib.parse.urlparse(uri)
    query_params = parse_unique_urlencoded(parts.query)
    query_params.update(params)
    new_query = urllib.parse.urlencode(query_params)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)