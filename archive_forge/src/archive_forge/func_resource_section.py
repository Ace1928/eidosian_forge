import os
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.exceptions import DataNotFoundError
import boto3
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter, ServiceResourceDocumenter
from boto3.utils import ServiceContext
def resource_section(self, section):
    section.style.h2('Resources')
    section.style.new_line()
    section.write('Resources are available in boto3 via the ``resource`` method. For more detailed instructions and examples on the usage of resources, see the resources ')
    section.style.external_link(title='user guide', link=self._USER_GUIDE_LINK)
    section.write('.')
    section.style.new_line()
    section.style.new_line()
    section.write('The available resources are:')
    section.style.new_line()
    section.style.toctree()
    self._document_service_resource(section)
    self._document_resources(section)