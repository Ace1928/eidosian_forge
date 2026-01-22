import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def convert_to_hostname(url: str) -> str:
    return url.replace('http://', '').replace('https://', '').split('/', 1)[0]