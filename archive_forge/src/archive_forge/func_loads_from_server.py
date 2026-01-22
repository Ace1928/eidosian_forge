import io
from typing import NamedTuple
from typing import Any
from typing import Dict
from typing import Optional
import ray.cloudpickle as cloudpickle
from ray.util.client import RayAPIStub
from ray.util.client.common import ClientObjectRef
from ray.util.client.common import ClientActorHandle
from ray.util.client.common import ClientActorRef
from ray.util.client.common import ClientActorClass
from ray.util.client.common import ClientRemoteFunc
from ray.util.client.common import ClientRemoteMethod
from ray.util.client.common import OptionWrapper
from ray.util.client.common import InProgressSentinel
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import pickle  # noqa: F401
def loads_from_server(data: bytes, *, fix_imports=True, encoding='ASCII', errors='strict') -> Any:
    if isinstance(data, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(data)
    return ServerUnpickler(file, fix_imports=fix_imports, encoding=encoding, errors=errors).load()