import io
import ray
from typing import Any
from typing import TYPE_CHECKING
from ray._private.client_mode_hook import disable_client_hook
import ray.cloudpickle as cloudpickle
from ray.util.client.client_pickler import PickleStub
from ray.util.client.server.server_stubs import ClientReferenceActor
from ray.util.client.server.server_stubs import ClientReferenceFunction
import pickle  # noqa: F401
def loads_from_client(data: bytes, server_instance: 'RayletServicer', *, fix_imports=True, encoding='ASCII', errors='strict') -> Any:
    with disable_client_hook():
        if isinstance(data, str):
            raise TypeError("Can't load pickle from unicode string")
        file = io.BytesIO(data)
        return ClientUnpickler(server_instance, file, fix_imports=fix_imports, encoding=encoding).load()