from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..message import Empty, Message
 Create an ``PULL-DOC-REQ`` message

        Any keyword arguments will be put into the message ``metadata``
        fragment as-is.

        