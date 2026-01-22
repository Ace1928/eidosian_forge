from __future__ import annotations
import errno
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Final
import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.httpserver import HTTPServer
from streamlit import cli_util, config, file_util, source_util, util
from streamlit.components.v1.components import ComponentRegistry
from streamlit.config_option import ConfigOption
from streamlit.logger import get_logger
from streamlit.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.runtime_util import get_max_message_size_bytes
from streamlit.web.cache_storage_manager_config import (
from streamlit.web.server.app_static_file_handler import AppStaticFileHandler
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
from streamlit.web.server.component_request_handler import ComponentRequestHandler
from streamlit.web.server.media_file_handler import MediaFileHandler
from streamlit.web.server.routes import (
from streamlit.web.server.server_util import DEVELOPMENT_PORT, make_url_path_regex
from streamlit.web.server.stats_request_handler import StatsRequestHandler
from streamlit.web.server.upload_file_request_handler import UploadFileRequestHandler
def start_listening_tcp_socket(http_server: HTTPServer) -> None:
    call_count = 0
    port = None
    while call_count < MAX_PORT_SEARCH_RETRIES:
        address = config.get_option('server.address')
        port = config.get_option('server.port')
        if int(port) == DEVELOPMENT_PORT:
            _LOGGER.warning('Port %s is reserved for internal development. It is strongly recommended to select an alternative port for `server.port`.', DEVELOPMENT_PORT)
        try:
            http_server.listen(port, address)
            break
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                if server_port_is_manually_set():
                    _LOGGER.error('Port %s is already in use', port)
                    sys.exit(1)
                else:
                    _LOGGER.debug('Port %s already in use, trying to use the next one.', port)
                    port += 1
                    if port == DEVELOPMENT_PORT:
                        port += 1
                    config.set_option('server.port', port, ConfigOption.STREAMLIT_DEFINITION)
                    call_count += 1
            else:
                raise
    if call_count >= MAX_PORT_SEARCH_RETRIES:
        raise RetriesExceeded(f'Cannot start Streamlit server. Port {port} is already in use, and Streamlit was unable to find a free port after {MAX_PORT_SEARCH_RETRIES} attempts.')