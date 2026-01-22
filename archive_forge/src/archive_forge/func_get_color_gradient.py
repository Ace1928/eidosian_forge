from __future__ import annotations
import ast
import csv
import inspect
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional
import numpy as np
import PIL
import PIL.Image
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import components, oauth, processing_utils, routes, utils, wasm_utils
from gradio.context import Context, LocalContext
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import EventData
from gradio.exceptions import Error
from gradio.flagging import CSVLogger
def get_color_gradient(c1, c2, n):
    if n < 1:
        raise ValueError('Must have at least one stop in gradient')
    c1_rgb = np.array(hex_to_rgb(c1)) / 255
    c2_rgb = np.array(hex_to_rgb(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [(1 - mix) * c1_rgb + mix * c2_rgb for mix in mix_pcts]
    return ['#' + ''.join((f'{int(round(val * 255)):02x}' for val in item)) for item in rgb_colors]