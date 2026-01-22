import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
@property
def cred_helpers(self) -> Dict:
    return self.get('credHelpers', {})