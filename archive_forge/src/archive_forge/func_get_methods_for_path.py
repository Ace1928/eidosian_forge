from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
def get_methods_for_path(self, path: str) -> List[str]:
    """Return a list of valid methods for the specified path."""
    from openapi_pydantic import Operation
    path_item = self._get_path_strict(path)
    results = []
    for method in HTTPVerb:
        operation = getattr(path_item, method.value, None)
        if isinstance(operation, Operation):
            results.append(method.value)
    return results