from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def list_instance_type(self, location=None):
    return [self._instance_type_to_size(instance) for name, instance in INSTANCE_TYPES.items()]