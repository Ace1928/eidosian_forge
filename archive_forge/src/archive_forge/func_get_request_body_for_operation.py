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
def get_request_body_for_operation(self, operation: Operation) -> Optional[RequestBody]:
    """Get the request body for a given operation."""
    from openapi_pydantic import Reference
    request_body = operation.requestBody
    if isinstance(request_body, Reference):
        request_body = self._get_root_referenced_request_body(request_body)
    return request_body