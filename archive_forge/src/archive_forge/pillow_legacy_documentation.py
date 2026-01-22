import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
Use Orientation information from EXIF meta data to
            orient the image correctly. Similar code as in FreeImage plugin.
            