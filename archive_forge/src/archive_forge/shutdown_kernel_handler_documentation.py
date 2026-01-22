import tornado
from jupyter_server.base.handlers import APIHandler
from nbclient.util import ensure_async
 Handler to shut down kernel on page's `beforeunload` event.
    