from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
Translate a schema.

        Parameters
        ----------
        schema: dict
            The schema to be translated

        Returns
        -------
        Dict
            The translated schema
        