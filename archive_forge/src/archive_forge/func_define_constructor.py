from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def define_constructor(py_class, struct_typeclass, fields):
    """Define the jit-code constructor for `struct_typeclass` using the
    Python type `py_class` and the required `fields`.

    Use this instead of `define_proxy()` if the user does not want boxing
    logic defined.
    """
    params = ', '.join(fields)
    indent = ' ' * 8
    init_fields_buf = []
    for k in fields:
        init_fields_buf.append(f'st.{k} = {k}')
    init_fields = f'\n{indent}'.join(init_fields_buf)
    source = f'\ndef ctor({params}):\n    struct_type = struct_typeclass(list(zip({list(fields)}, [{params}])))\n    def impl({params}):\n        st = new(struct_type)\n        {init_fields}\n        return st\n    return impl\n'
    glbs = dict(struct_typeclass=struct_typeclass, new=new)
    exec(source, glbs)
    ctor = glbs['ctor']
    overload(py_class)(ctor)