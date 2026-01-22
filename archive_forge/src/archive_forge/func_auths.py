import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
@property
def auths(self) -> Dict[str, Dict[str, Any]]:
    return self.get('auths', {})