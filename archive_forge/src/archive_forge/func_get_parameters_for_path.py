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
def get_parameters_for_path(self, path: str) -> List[Parameter]:
    from openapi_pydantic import Reference
    path_item = self._get_path_strict(path)
    parameters = []
    if not path_item.parameters:
        return []
    for parameter in path_item.parameters:
        if isinstance(parameter, Reference):
            parameter = self._get_root_referenced_parameter(parameter)
        parameters.append(parameter)
    return parameters