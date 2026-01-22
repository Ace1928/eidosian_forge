from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_shape_default(self, section, shape, history, include=None, exclude=None, **kwargs):
    self._add_member_documentation(section, shape, **kwargs)