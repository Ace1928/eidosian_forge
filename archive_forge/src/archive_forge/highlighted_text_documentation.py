from __future__ import annotations
from typing import Any, Callable, List, Union
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import Events

        Parameters:
            value: Expects a list of (word, category) tuples, or a dictionary of two keys: "text", and "entities", which itself is a list of dictionaries, each of which have the keys: "entity" (or "entity_group"), "start", and "end"
        Returns:
            An instance of HighlightedTextData
        