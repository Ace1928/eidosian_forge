import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def import_image_from_stream(self, stream, repository=None, tag=None, changes=None):
    return self.import_image(src=stream, stream_src=True, repository=repository, tag=tag, changes=changes)