from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ...core.serialization import Serializer
from ...document.callbacks import invoke_with_curdoc
from ...document.json import PatchJson
from ..message import Message


        