import weakref
import importlib_metadata
from wsme.exc import ClientSideError
def register_protocol(protocol):
    registered_protocols[protocol.name] = protocol