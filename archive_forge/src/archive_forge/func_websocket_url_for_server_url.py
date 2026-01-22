from __future__ import annotations
import logging # isort:skip
def websocket_url_for_server_url(url: str) -> str:
    """ Convert an ``http(s)`` URL for a Bokeh server websocket endpoint into
    the appropriate ``ws(s)`` URL

    Args:
        url (str):
            An ``http(s)`` URL

    Returns:
        str:
            The corresponding ``ws(s)`` URL ending in ``/ws``

    Raises:
        ValueError:
            If the input URL is not of the proper form.

    """
    if url.startswith('http:'):
        reprotocoled = 'ws' + url[4:]
    elif url.startswith('https:'):
        reprotocoled = 'wss' + url[5:]
    else:
        raise ValueError('URL has unknown protocol ' + url)
    if reprotocoled.endswith('/'):
        return reprotocoled + 'ws'
    else:
        return reprotocoled + '/ws'