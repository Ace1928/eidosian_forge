from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
def define_proxy(py_class, struct_typeclass, fields):
    """Defines a PyObject proxy for a structref.

    This makes `py_class` a valid constructor for creating a instance of
    `struct_typeclass` that contains the members as defined by `fields`.

    Parameters
    ----------
    py_class : type
        The Python class for constructing an instance of `struct_typeclass`.
    struct_typeclass : numba.core.types.Type
        The structref type class to bind to.
    fields : Sequence[str]
        A sequence of field names.

    Returns
    -------
    None
    """
    define_constructor(py_class, struct_typeclass, fields)
    define_boxing(struct_typeclass, py_class)