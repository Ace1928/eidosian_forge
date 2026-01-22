from __future__ import annotations
import logging # isort:skip
def server_url_for_websocket_url(url: str) -> str:
    """ Convert an ``ws(s)`` URL for a Bokeh server into the appropriate
    ``http(s)`` URL for the websocket endpoint.

    Args:
        url (str):
            An ``ws(s)`` URL ending in ``/ws``

    Returns:
        str:
            The corresponding ``http(s)`` URL.

    Raises:
        ValueError:
            If the input URL is not of the proper form.

    """
    if url.startswith('ws:'):
        reprotocoled = 'http' + url[2:]
    elif url.startswith('wss:'):
        reprotocoled = 'https' + url[3:]
    else:
        raise ValueError('URL has non-websocket protocol ' + url)
    if not reprotocoled.endswith('/ws'):
        raise ValueError('websocket URL does not end in /ws')
    return reprotocoled[:-2]