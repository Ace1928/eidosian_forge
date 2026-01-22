from botocore.docs.params import ResponseParamsDocumenter
from boto3.docs.utils import get_identifier_description
def document_reference(section, reference_model, include_signature=True):
    if include_signature:
        section.style.start_sphinx_py_attr(reference_model.name)
    reference_type = '(:py:class:`%s`) ' % reference_model.resource.type
    section.write(reference_type)
    section.include_doc_string('The related %s if set, otherwise ``None``.' % reference_model.name)