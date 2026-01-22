from botocore.docs.params import ResponseParamsDocumenter
from boto3.docs.utils import get_identifier_description
def document_attribute(section, service_name, resource_name, attr_name, event_emitter, attr_model, include_signature=True):
    if include_signature:
        section.style.start_sphinx_py_attr(attr_name)
    ResourceShapeDocumenter(service_name=service_name, operation_name=resource_name, event_emitter=event_emitter).document_params(section=section, shape=attr_model)