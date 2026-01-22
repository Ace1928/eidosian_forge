from __future__ import annotations
import json
from typing import TYPE_CHECKING
import pyarrow
from pandas.compat import pa_version_under14p1
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.interval import VALID_CLOSED
def patch_pyarrow():
    if not pa_version_under14p1:
        return
    if getattr(pyarrow, '_hotfix_installed', False):
        return

    class ForbiddenExtensionType(pyarrow.ExtensionType):

        def __arrow_ext_serialize__(self):
            return b''

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            import io
            import pickletools
            out = io.StringIO()
            pickletools.dis(serialized, out)
            raise RuntimeError(_ERROR_MSG.format(storage_type=storage_type, serialized=serialized, pickle_disassembly=out.getvalue()))
    pyarrow.unregister_extension_type('arrow.py_extension_type')
    pyarrow.register_extension_type(ForbiddenExtensionType(pyarrow.null(), 'arrow.py_extension_type'))
    pyarrow._hotfix_installed = True