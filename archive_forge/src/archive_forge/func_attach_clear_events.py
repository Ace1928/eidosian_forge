from __future__ import annotations
import inspect
import json
import os
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio import Examples, utils
from gradio.blocks import Blocks
from gradio.components import (
from gradio.data_classes import InterfaceTypes
from gradio.events import Dependency, Events, on
from gradio.exceptions import RenderError
from gradio.flagging import CSVLogger, FlaggingCallback, FlagMethod
from gradio.layouts import Accordion, Column, Row, Tab, Tabs
from gradio.pipelines import load_from_pipeline
from gradio.themes import ThemeClass as Theme
def attach_clear_events(self, _clear_btn: ClearButton, input_component_column: Column | None):
    _clear_btn.add(self.input_components + self.output_components)
    _clear_btn.click(None, [], [input_component_column] if input_component_column else [], js=f'() => {json.dumps([{'variant': None, 'visible': True, '__type__': 'update'}] if self.interface_type in [InterfaceTypes.STANDARD, InterfaceTypes.INPUT_ONLY, InterfaceTypes.UNIFIED] else [])}\n            ')