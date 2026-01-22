import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
@ray.remote
def calculate_identifier(obj: Any) -> str:
    """Calculate a url-safe identifier for an object."""
    m = hashlib.sha256()
    m.update(cloudpickle.dumps(obj))
    hash = m.digest()
    encoded = base64.urlsafe_b64encode(hash).decode('ascii')
    return encoded