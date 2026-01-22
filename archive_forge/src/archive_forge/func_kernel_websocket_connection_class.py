from jupyter_core.utils import ensure_async
from tornado import web
from tornado.websocket import WebSocketHandler
from jupyter_server.auth.decorator import ws_authenticated
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.base.websocket import WebSocketMixin
@property
def kernel_websocket_connection_class(self):
    """The kernel websocket connection class."""
    return self.settings.get('kernel_websocket_connection_class')