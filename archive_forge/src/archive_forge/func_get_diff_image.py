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
def get_diff_image(self):
    if self._png_is_old:
        renderer = self.get_renderer()
        pixels = np.asarray(renderer.buffer_rgba())
        buff = pixels.view(np.uint32).squeeze(2)
        if self._force_full or buff.shape != self._last_buff.shape or (pixels[:, :, 3] != 255).any():
            self.set_image_mode('full')
            output = buff
        else:
            self.set_image_mode('diff')
            diff = buff != self._last_buff
            output = np.where(diff, buff, 0)
        self._last_buff = buff.copy()
        self._force_full = False
        self._png_is_old = False
        data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
        with BytesIO() as png:
            Image.fromarray(data).save(png, format='png')
            return png.getvalue()