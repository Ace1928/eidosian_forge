import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, NamedTuple, Optional
from ray.actor import ActorHandle
from ray.serve.generated.serve_pb2 import ApplicationStatus as ApplicationStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import DeploymentStatus as DeploymentStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
@classmethod
def from_replica_tag(cls, tag):
    parsed = tag.split(cls.delimiter)
    if len(parsed) == 3:
        return cls(app_name=parsed[0], deployment_name=parsed[1], replica_suffix=parsed[2])
    elif len(parsed) == 2:
        return cls('', deployment_name=parsed[0], replica_suffix=parsed[1])
    else:
        raise ValueError(f"Given replica tag {tag} didn't match pattern, please ensure it has either two or three fields with delimiter {cls.delimiter}")