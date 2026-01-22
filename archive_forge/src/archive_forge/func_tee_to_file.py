import io
def tee_to_file(response, filename, chunksize=_DEFAULT_CHUNKSIZE, decode_content=None):
    """Stream the response both to the generator and a file.

    This will open a file named ``filename`` and stream the response body
    while writing the bytes to the opened file object.

    Example usage:

    .. code-block:: python

        resp = requests.get(url, stream=True)
        for chunk in tee_to_file(resp, 'save_file'):
            # do stuff with chunk

    :param response: Response from requests.
    :type response: requests.Response
    :param str filename: Name of file in which we write the response content.
    :param int chunksize: (optional), Size of chunk to attempt to stream.
    :param bool decode_content: (optional), If True, this will decode the
        compressed content of the response.
    """
    with open(filename, 'wb') as fd:
        for chunk in tee(response, fd, chunksize, decode_content):
            yield chunk