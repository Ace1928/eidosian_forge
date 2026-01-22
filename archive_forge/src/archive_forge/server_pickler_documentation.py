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
Implements the client side of the client/server pickling protocol.

These picklers are aware of the server internals and can find the
references held for the client within the server.

More discussion about the client/server pickling protocol can be found in:

  ray/util/client/client_pickler.py

ServerPickler dumps ray objects from the server into the appropriate stubs.
ClientUnpickler loads stubs from the client and finds their associated handle
in the server instance.
