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
def requires_actor_restart(self, new_version):
    """Determines whether the new version requires actors of the current version to
        be restarted.
        """
    return self.code_version != new_version.code_version or self.ray_actor_options_hash != new_version.ray_actor_options_hash or self.placement_group_options_hash != new_version.placement_group_options_hash or (self.max_replicas_per_node != new_version.max_replicas_per_node)