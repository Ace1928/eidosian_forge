import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
@property
def flash_version(self) -> str:
    return self.metadata['version'].get('flash', '0.0.0')