from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
def iter_versions(self):
    for verneed, vernaux in super(GNUVerNeedSection, self).iter_versions():
        verneed.name = self.stringtable.get_string(verneed['vn_file'])
        yield (verneed, vernaux)