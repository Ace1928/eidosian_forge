from __future__ import annotations
from ..runtime import driver
def get_address_tma_mapping(self):
    return {self.globalAddressArgIdx: self.TMADescArgIdx + len(self.ids_of_folded_args)}