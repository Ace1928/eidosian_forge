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
def handle_refresh(self, event):
    figure_label = self.figure.get_label()
    if not figure_label:
        figure_label = f'Figure {self.manager.num}'
    self.send_event('figure_label', label=figure_label)
    self._force_full = True
    if self.toolbar:
        self.toolbar.set_history_buttons()
    self.draw_idle()