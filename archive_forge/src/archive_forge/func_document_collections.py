from botocore import xform_name
from botocore.docs.method import get_instance_public_methods
from botocore.docs.utils import DocumentedShape
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import add_resource_type_overview
def document_collections(self, section):
    collections = self._resource.meta.resource_model.collections
    collections_list = []
    add_resource_type_overview(section=section, resource_type='Collections', description='Collections provide an interface to iterate over and manipulate groups of resources. ', intro_link='guide_collections')
    self.member_map['collections'] = collections_list
    for collection in collections:
        collection_section = section.add_new_section(collection.name)
        collections_list.append(collection.name)
        self._document_collection(collection_section, collection)