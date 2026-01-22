from pyomo.core.base.reference import Reference
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
def rename_components(model, component_list, prefix):
    """
    Rename components in component_list using the prefix AND
    unique_component_name

    Parameters
    ----------
    model : Pyomo model (or Block)
       The variables, constraints and objective will be renamed on this model
    component_list : list
       List of components to rename
    prefix : str
       The prefix to use when building the new names

    Examples
    --------
    >>> c_list = list(model.component_objects(ctype=Var, descend_into=True))
    >>> rename_components(model, component_list=c_list, prefix='special_')

    Returns
    -------
    ComponentMap : maps the renamed Component objects
       to a string that provides their old fully qualified names

    ToDo
    ----
    - need to add a check to see if someone accidentally passes a generator since this can lead to an infinite loop

    """
    refs = {}
    for c in component_list:
        if c.is_reference():
            refs[c] = {}
            for k, v in c._data.items():
                refs[c][k] = (v.parent_block(), v.local_name)
    name_map = ComponentMap()
    for c in component_list:
        if not c.is_reference():
            parent = c.parent_block()
            old_name = c.name
            new_name = unique_component_name(parent, prefix + c.local_name)
            parent.del_component(c)
            parent.add_component(new_name, c)
            name_map[c] = old_name
    for c in refs:
        new_map = ComponentMap()
        for k, v in refs[c].items():
            new_data = v[0].find_component(prefix + v[1])
            if new_data is None:
                new_data = v[0].find_component(v[1])
            if new_data is None:
                raise RuntimeError(f'Unable to remap Reference {c.name} whilst renaming components.')
            new_map[k] = new_data
        parent = c.parent_block()
        old_name = c.name
        new_name = unique_component_name(parent, prefix + c.local_name)
        parent.del_component(c)
        cnew = Reference(new_map)
        parent.add_component(new_name, cnew)
        name_map[cnew] = old_name
    return name_map