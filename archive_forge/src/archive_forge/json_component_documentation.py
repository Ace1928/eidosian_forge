from __future__ import annotations
import json
from typing import Any, Callable
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.events import Events

        Parameters:
            value: Expects a `str` filepath to a file containing valid JSON -- or a `list` or `dict` that is valid JSON
        Returns:
            Returns the JSON as a `list` or `dict`.
        