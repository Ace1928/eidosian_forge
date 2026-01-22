import os
from boto3.docs.service import ServiceDocumenter
def generate_docs(root_dir, session):
    """Generates the reference documentation for botocore

    This will go through every available AWS service and output ReSTructured
    text files documenting each service.

    :param root_dir: The directory to write the reference files to. Each
        service's reference documentation is loacated at
        root_dir/reference/services/service-name.rst

    :param session: The boto3 session
    """
    services_doc_path = os.path.join(root_dir, 'reference', 'services')
    if not os.path.exists(services_doc_path):
        os.makedirs(services_doc_path)
    for service_name in session.get_available_services():
        docs = ServiceDocumenter(service_name, session).document_service()
        service_doc_path = os.path.join(services_doc_path, service_name + '.rst')
        with open(service_doc_path, 'wb') as f:
            f.write(docs)