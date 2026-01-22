from __future__ import annotations
import abc
import hashlib
import json
import sys
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import gradio_client.utils as client_utils
from gradio import utils
from gradio.blocks import Block, BlockContext
from gradio.component_meta import ComponentMeta
from gradio.data_classes import GradioDataModel
from gradio.events import EventListener
from gradio.layouts import Form
from gradio.processing_utils import move_files_to_cache
@classmethod
def has_event(cls, event: str | EventListener) -> bool:
    return event in cls.EVENTS