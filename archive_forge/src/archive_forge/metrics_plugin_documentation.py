import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
Gets the image data for a blob key.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            blob_key: a string identifier for a DataProvider blob.

        Returns:
            A tuple containing:
              data: a raw bytestring of the requested image's contents.
              content_type: a string HTTP content type.
        