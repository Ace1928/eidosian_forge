from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
@classmethod
def validate_monty_v2(cls, __input_value, _):
    """
        Pydantic validator with correct signature for pydantic v2.x
        """
    return cls._validate_monty(__input_value)