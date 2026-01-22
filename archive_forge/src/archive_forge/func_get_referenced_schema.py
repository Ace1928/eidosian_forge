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
def get_referenced_schema(self, ref: Reference) -> Schema:
    """Get a schema (or nested reference) or err."""
    ref_name = ref.ref.split('/')[-1]
    schemas = self._schemas_strict
    if ref_name not in schemas:
        raise ValueError(f'No schema found for {ref_name}')
    return schemas[ref_name]