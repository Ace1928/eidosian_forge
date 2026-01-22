from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_shape_type_structure(self, section, shape, history, include=None, exclude=None, **kwargs):
    if len(history) > 1:
        self._add_member_documentation(section, shape, **kwargs)
        section.style.indent()
    members = self._add_members_to_shape(shape.members, include)
    for i, param in enumerate(members):
        if exclude and param in exclude:
            continue
        param_shape = members[param]
        param_section = section.add_new_section(param, context={'shape': param_shape.name})
        param_section.style.new_line()
        is_required = param in shape.required_members
        self.traverse_and_document_shape(section=param_section, shape=param_shape, history=history, name=param, is_required=is_required)
    section = section.add_new_section('end-structure')
    if len(history) > 1:
        section.style.dedent()
    section.style.new_line()