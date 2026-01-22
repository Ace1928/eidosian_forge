import ast
import re
from collections import OrderedDict
def obj_from_atoms(atoms, namespace):
    """Return object defined by list `atoms` in dict-like `namespace`

    Parameters
    ----------
    atoms : list
        List of :class:`atoms`
    namespace : dict-like
        Namespace in which object will be defined.

    Returns
    -------
    obj_root : object
        Namespace such that we can set a desired value to the object defined in
        `atoms` with ``obj_root[obj_key] = value``.
    obj_key : str or int
        Index into list or key into dictionary for `obj_root`.
    """
    root_obj = namespace
    for el in atoms:
        prev_root = root_obj
        if isinstance(el.op, (ast.Attribute, ast.Name)):
            root_obj = _create_obj_in(el, root_obj)
        else:
            root_obj = _create_subscript_in(el, root_obj)
        if not isinstance(root_obj, el.obj_type):
            raise AscconvParseError(f'Unexpected type for {el.obj_id} in {prev_root}')
    return (prev_root, el.obj_id)