from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4
import urllib.parse
from langsmith import schemas as ls_schemas
from langsmith import utils
from langsmith.client import ID_TYPE, RUN_TYPE_T, Client, _dumps_json
@classmethod
def from_dotted_order(cls, dotted_order: str, **kwargs: Any) -> RunTree:
    """Create a new 'child' span from the provided dotted order.

        Returns:
            RunTree: The new span.
        """
    headers = {f'{LANGSMITH_DOTTED_ORDER}': dotted_order}
    return cast(RunTree, cls.from_headers(headers, **kwargs))