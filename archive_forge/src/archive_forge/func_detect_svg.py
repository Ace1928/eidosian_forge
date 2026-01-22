import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def detect_svg(data, f):
    del f
    if data.startswith(b'<?xml ') or data.startswith(b'<svg '):
        return 'svg'