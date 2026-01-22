import os
import boto3
from botocore.exceptions import DataNotFoundError
from botocore.docs.service import ServiceDocumenter as BaseServiceDocumenter
from botocore.docs.bcdoc.restdoc import DocumentStructure
from boto3.utils import ServiceContext
from boto3.docs.client import Boto3ClientDocumenter
from boto3.docs.resource import ResourceDocumenter
from boto3.docs.resource import ServiceResourceDocumenter
def client_api(self, section):
    examples = None
    try:
        examples = self.get_examples(self._service_name)
    except DataNotFoundError:
        pass
    Boto3ClientDocumenter(self._client, examples).document_client(section)