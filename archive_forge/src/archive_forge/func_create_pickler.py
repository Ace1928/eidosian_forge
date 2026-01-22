from pickle import (  # type: ignore[attr-defined]
from struct import pack
from types import FunctionType
from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer
def create_pickler(data_buf, importer, protocol=4):
    if importer is sys_importer:
        return Pickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)