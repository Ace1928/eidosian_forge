import base64
import os
from anyio.to_thread import run_sync
from tornado import web
from jupyter_server.services.contents.filemanager import (
Save content of a generic file.