import asyncio
import json
import os
import threading
import warnings
from enum import Enum
from functools import partial
from typing import Awaitable, Dict
import websockets
from jupyterlab_server.themes_handler import ThemesHandler
from markupsafe import Markup
from nbconvert.exporters.html import find_lab_theme
from .static_file_handler import TemplateStaticFileHandler
def wait_for_request(url: str=None) -> str:
    """Helper function to pause the execution of notebook and wait for
    the pre-heated kernel to be used and all request info is added to
    the environment.

    Args:
        url (str, optional): Address to get request info, if it is not
        provided, `voila` will figure out from the environment variables.
        Defaults to None.

    """
    preheat_mode = os.getenv(ENV_VARIABLE.VOILA_PREHEAT, 'False')
    if preheat_mode == 'False':
        return
    request_info = None
    if url is None:
        protocol = os.getenv(ENV_VARIABLE.VOILA_WS_PROTOCOL, 'ws')
        server_ip = os.getenv(ENV_VARIABLE.VOILA_APP_IP, '127.0.0.1')
        server_port = os.getenv(ENV_VARIABLE.VOILA_APP_PORT, '8866')
        server_url = os.getenv(ENV_VARIABLE.VOILA_SERVER_URL, '/')
        ws_base_url = os.getenv(ENV_VARIABLE.VOILA_WS_BASE_URL, server_url)
        url = f'{protocol}://{server_ip}:{server_port}{ws_base_url}voila/query'
    kernel_id = os.getenv(ENV_VARIABLE.VOILA_KERNEL_ID)
    ws_url = f'{url}/{kernel_id}'

    def inner():
        nonlocal request_info
        loop = asyncio.new_event_loop()
        request_info = loop.run_until_complete(_get_request_info(ws_url))
    thread = threading.Thread(target=inner)
    try:
        thread.start()
        thread.join()
    except (KeyboardInterrupt, SystemExit):
        asyncio.get_event_loop().stop()
    if request_info is not None:
        for k, v in json.loads(request_info).items():
            os.environ[k] = v