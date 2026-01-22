import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
def handle_set_dpi_ratio(self, event):
    self._handle_set_device_pixel_ratio(event.get('dpi_ratio', 1))