from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def simply_type(result_type):
    result_type = PyrexTypes.remove_cv_ref(result_type, remove_fakeref=True)
    if result_type.is_array:
        result_type = PyrexTypes.c_ptr_type(result_type.base_type)
    return result_type