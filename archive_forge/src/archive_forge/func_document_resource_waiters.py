from botocore import xform_name
from botocore.utils import get_service_module_name
from botocore.docs.method import document_model_driven_method
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.utils import add_resource_type_overview
def document_resource_waiters(self, section):
    waiters = self._resource.meta.resource_model.waiters
    add_resource_type_overview(section=section, resource_type='Waiters', description='Waiters provide an interface to wait for a resource to reach a specific state.', intro_link='waiters_intro')
    waiter_list = []
    self.member_map['waiters'] = waiter_list
    for waiter in waiters:
        waiter_section = section.add_new_section(waiter.name)
        waiter_list.append(waiter.name)
        document_resource_waiter(section=waiter_section, resource_name=self._resource_name, event_emitter=self._resource.meta.client.meta.events, service_model=self._service_model, resource_waiter_model=waiter, service_waiter_model=self._service_waiter_model)