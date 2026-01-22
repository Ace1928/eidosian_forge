import collections
from requests import compat
def dump_response(response, request_prefix=b'< ', response_prefix=b'> ', data_array=None):
    """Dump a single request-response cycle's information.

    This will take a response object and dump only the data that requests can
    see for that single request-response cycle.

    Example::

        import requests
        from requests_toolbelt.utils import dump

        resp = requests.get('https://api.github.com/users/sigmavirus24')
        data = dump.dump_response(resp)
        print(data.decode('utf-8'))

    :param response:
        The response to format
    :type response: :class:`requests.Response`
    :param request_prefix: (*optional*)
        Bytes to prefix each line of the request data
    :type request_prefix: :class:`bytes`
    :param response_prefix: (*optional*)
        Bytes to prefix each line of the response data
    :type response_prefix: :class:`bytes`
    :param data_array: (*optional*)
        Bytearray to which we append the request-response cycle data
    :type data_array: :class:`bytearray`
    :returns: Formatted bytes of request and response information.
    :rtype: :class:`bytearray`
    """
    data = data_array if data_array is not None else bytearray()
    prefixes = PrefixSettings(request_prefix, response_prefix)
    if not hasattr(response, 'request'):
        raise ValueError('Response has no associated request')
    proxy_info = _get_proxy_information(response)
    _dump_request_data(response.request, prefixes, data, proxy_info=proxy_info)
    _dump_response_data(response, prefixes, data)
    return data