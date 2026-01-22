from typing import TYPE_CHECKING
from types import SimpleNamespace
def build_runtime_context(self) -> 'RuntimeContext':
    """Creates a RuntimeContext backed by the properites of this API"""
    from ray.runtime_context import RuntimeContext
    return RuntimeContext(self)