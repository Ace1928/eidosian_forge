from ray.core.generated.common_pb2 import (
from ray.core.generated.gcs_pb2 import (
from ray.dashboard.memory_utils import ReferenceType
from typing import Literal
def validate_protobuf_enum(grpc_enum, custom_enum):
    """Validate the literal contains the correct enum values from protobuf"""
    enum_vals = set(grpc_enum.DESCRIPTOR.values_by_name)
    if len(enum_vals) > 0:
        assert enum_vals == set(custom_enum)