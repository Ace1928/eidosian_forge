from __future__ import annotations
import warnings
from typing import Any, Callable
from gradio.components.base import FormComponent
from gradio.events import Events

        Parameters:
            value: Expects a `str | int | float` corresponding to the value of the dropdown entry to be selected.
        Returns:
            Returns the value of the selected dropdown entry.
        