import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import deprecated
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.container_utils import define_homogeneous_container_type
def local_suffix_generator(blk, datatype=_noarg, active=True, descend_into=True):
    """
    Generates an efficient traversal of all suffixes that
    have been declared local data storage.

    Args:
        blk: A block object.
        datatype: Restricts the suffixes included in the
            returned generator to those matching the
            provided suffix datatype.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of suffixes
    """
    for suf in filter(lambda x: x.direction is suffix.LOCAL and (datatype is _noarg or x.datatype is datatype), blk.components(ctype=suffix._ctype, active=active, descend_into=descend_into)):
        yield suf