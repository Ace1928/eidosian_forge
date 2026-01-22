from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_params(self, section, shape, include=None, exclude=None):
    """Fills out the documentation for a section given a model shape.

        :param section: The section to write the documentation to.

        :param shape: The shape of the operation.

        :type include: Dictionary where keys are parameter names and
            values are the shapes of the parameter names.
        :param include: The parameter shapes to include in the documentation.

        :type exclude: List of the names of the parameters to exclude.
        :param exclude: The names of the parameters to exclude from
            documentation.
        """
    history = []
    self.traverse_and_document_shape(section=section, shape=shape, history=history, name=None, include=include, exclude=exclude)