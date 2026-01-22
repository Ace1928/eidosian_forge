from __future__ import annotations
from ..runtime import driver
def parse_tma_info(infos, ids_of_folded_args):
    ret = []
    for info in infos:
        e = InfoFromBackendForTensorMap(infos=info)
        e.ids_of_folded_args = ids_of_folded_args
        ret.append(e)
    return ret