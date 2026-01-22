from __future__ import annotations
from pathlib import Path
from typing import Any
from gradio_client import file
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import FileData
from gradio.events import Events

        Parameters:
            value: Expects a `str` or `pathlib.Path` object containing the path to the image.
        Returns:
            A FileData object containing the image data.
        