from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def show_app(app: Application, state: State, notebook_url: str | ProxyUrlFunc=DEFAULT_JUPYTER_URL, port: int=0, **kw: Any) -> None:
    """ Embed a Bokeh server application in a Jupyter Notebook output cell.

    Args:
        app (Application or callable) :
            A Bokeh Application to embed inline in a Jupyter notebook.

        state (State) :
            ** Unused **

        notebook_url (str or callable) :
            The URL of the notebook server that is running the embedded app.

            If ``notebook_url`` is a string, the value string is parsed to
            construct the origin and full server URLs.

            If notebook_url is a callable, it must accept one parameter,
            which will be the server port, or None. If passed a port,
            the callable must generate the server URL, otherwise if passed
            None, it must generate the origin URL for the server.

            If the environment variable JUPYTER_BOKEH_EXTERNAL_URL is set
            to the external URL of a JupyterHub, notebook_url is overridden
            with a callable which enables Bokeh to traverse the JupyterHub
            proxy without specifying this parameter.

        port (int) :
            A port for the embedded server will listen on.

            By default the port is 0, which results in the server listening
            on a random dynamic port.

    Any additional keyword arguments are passed to :class:`~bokeh.server.Server` (added in version 1.1)

    Returns:
        None

    """
    logging.basicConfig()
    from tornado.ioloop import IOLoop
    from ..server.server import Server
    loop = IOLoop.current()
    notebook_url = _update_notebook_url_from_env(notebook_url)
    if callable(notebook_url):
        origin = notebook_url(None)
    else:
        origin = _origin_url(notebook_url)
    server = Server({'/': app}, io_loop=loop, port=port, allow_websocket_origin=[origin], **kw)
    server_id = ID(uuid4().hex)
    curstate().uuid_to_server[server_id] = server
    server.start()
    if callable(notebook_url):
        url = notebook_url(server.port)
    else:
        url = _server_url(notebook_url, server.port)
    logging.debug(f'Server URL is {url}')
    logging.debug(f'Origin URL is {origin}')
    from ..embed import server_document
    script = server_document(url, resources=None)
    publish_display_data({HTML_MIME_TYPE: script, EXEC_MIME_TYPE: ''}, metadata={EXEC_MIME_TYPE: {'server_id': server_id}})