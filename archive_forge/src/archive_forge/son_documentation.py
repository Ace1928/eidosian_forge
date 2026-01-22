from __future__ import annotations
import copy
import re
from collections.abc import Mapping as _Mapping
from typing import (
Convert a SON document to a normal Python dictionary instance.

        This is trickier than just *dict(...)* because it needs to be
        recursive.
        