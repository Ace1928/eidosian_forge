import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        