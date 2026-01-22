import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def resolve_index_name(index_name: str) -> str:
    index_name = convert_to_hostname(index_name)
    if index_name == 'index.' + INDEX_NAME:
        index_name = INDEX_NAME
    return index_name