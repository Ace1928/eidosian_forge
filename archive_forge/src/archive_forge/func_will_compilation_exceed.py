import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def will_compilation_exceed(self, limit: int) -> bool:
    return self.will_compilation_exceed_bucket(limit) or self.will_compilation_exceed_total()