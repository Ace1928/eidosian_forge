import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def get_multivar(self, section: SectionLike, name: NameLike) -> Iterator[Value]:
    if not isinstance(section, tuple):
        section = (section,)
    for backend in self.backends:
        try:
            yield from backend.get_multivar(section, name)
        except KeyError:
            pass