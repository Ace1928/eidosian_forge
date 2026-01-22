from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.list_container import ListContainer
def define_simple_containers(namespace, prefix, ctype, use_slots=True):
    """Use this function to define all three simple
    container definitions for a new object category type."""
    doc_ = 'A %s-style container for objects with category type ' + ctype.__name__
    for suffix, container_class in (('tuple', TupleContainer), ('list', ListContainer), ('dict', DictContainer)):
        doc = doc_ % (suffix,)
        define_homogeneous_container_type(namespace, prefix + '_' + suffix, container_class, ctype, doc=doc, use_slots=use_slots)