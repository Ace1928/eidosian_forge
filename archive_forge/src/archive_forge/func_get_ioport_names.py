import importlib
import os
from .. import ports
def get_ioport_names(self, **kwargs):
    """Return a list of all I/O port names."""
    devices = self._get_devices(**self._add_api(kwargs))
    inputs = [device['name'] for device in devices if device['is_input']]
    outputs = {device['name'] for device in devices if device['is_output']}
    return [name for name in inputs if name in outputs]