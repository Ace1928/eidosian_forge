from botocore.docs.params import ResponseParamsDocumenter
from boto3.docs.utils import get_identifier_description
def document_identifier(section, resource_name, identifier_model, include_signature=True):
    if include_signature:
        section.style.start_sphinx_py_attr(identifier_model.name)
    description = get_identifier_description(resource_name, identifier_model.name)
    description = '*(string)* ' + description
    section.write(description)