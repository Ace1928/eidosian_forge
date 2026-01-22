from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
def document_shape_type_string(self, section, shape, history, include=None, exclude=None, **kwargs):
    if 'enum' in shape.metadata:
        for i, enum in enumerate(shape.metadata['enum']):
            section.write("'%s'" % enum)
            if i < len(shape.metadata['enum']) - 1:
                section.write('|')
    else:
        self.document_shape_default(section, shape, history)