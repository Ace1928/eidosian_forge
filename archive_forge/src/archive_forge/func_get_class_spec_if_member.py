from ..common.utils import bytes2str
def get_class_spec_if_member(func_spec, the_func):
    if 'DW_AT_object_pointer' in the_func.attributes:
        this_param = the_func.get_DIE_from_attribute('DW_AT_object_pointer')
        this_type = parse_cpp_datatype(this_param)
        class_spec = ClassDesc()
        class_spec.scopes = this_type.scopes + (this_type.name,)
        class_spec.const_member = any((('const', 'pointer') == this_type.modifiers[i:i + 2] for i in range(len(this_type.modifiers))))
        return class_spec
    parent = func_spec.get_parent()
    scopes = []
    while parent.tag in ('DW_TAG_class_type', 'DW_TAG_structure_type', 'DW_TAG_namespace'):
        scopes.insert(0, DIE_name(parent))
        parent = parent.get_parent()
    if scopes:
        cs = ClassDesc()
        cs.scopes = tuple(scopes)
        return cs
    return None