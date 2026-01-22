from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_shape_type_list(self, section, shape, history, include=None, exclude=None, **kwargs):
    self._add_member_documentation(section, shape, **kwargs)
    param_shape = shape.member
    param_section = section.add_new_section(param_shape.name, context={'shape': shape.member.name})
    self._start_nested_param(param_section)
    self.traverse_and_document_shape(section=param_section, shape=param_shape, history=history, name=None)
    section = section.add_new_section('end-list')
    self._end_nested_param(section)