from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def reinfer():
    dirty = False
    for entry in inferred:
        for assmt in entry.cf_assignments:
            assmt.infer_type()
        types = inferred_types(entry)
        new_type = spanning_type(types, entry.might_overflow, scope)
        if new_type != entry.type:
            self.set_entry_type(entry, new_type, scope)
            dirty = True
    return dirty