from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .singletons import Intrinsic

    A property which state won't be synced with the browser.
    