import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
@property
def build_env(self) -> Dict[str, Any]:
    return self.metadata['env']