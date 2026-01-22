from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def get_relocation(self, n):
    """ Get the relocation at index #n from the section (Relocation object)
        """
    if self._cached_relocations is None:
        self._cached_relocations = list(self.iter_relocations())
    return self._cached_relocations[n]