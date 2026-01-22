import json
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional
from zlib import crc32
from ray._private.pydantic_compat import BaseModel
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.utils import DeploymentOptionUpdateType, get_random_letters
from ray.serve.generated.serve_pb2 import DeploymentVersion as DeploymentVersionProto
def requires_actor_reconfigure(self, new_version):
    """Determines whether the new version requires calling reconfigure() on the
        replica actor.
        """
    return self.reconfigure_actor_hash != new_version.reconfigure_actor_hash