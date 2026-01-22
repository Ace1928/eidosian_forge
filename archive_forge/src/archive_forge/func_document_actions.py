from botocore import xform_name
from botocore.model import OperationModel
from botocore.utils import get_service_module_name
from botocore.docs.method import document_model_driven_method
from botocore.docs.method import document_custom_method
from boto3.docs.base import BaseDocumenter
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.utils import get_resource_public_actions
from boto3.docs.utils import add_resource_type_overview
def document_actions(self, section):
    modeled_actions_list = self._resource_model.actions
    modeled_actions = {}
    for modeled_action in modeled_actions_list:
        modeled_actions[modeled_action.name] = modeled_action
    resource_actions = get_resource_public_actions(self._resource.__class__)
    self.member_map['actions'] = sorted(resource_actions)
    add_resource_type_overview(section=section, resource_type='Actions', description='Actions call operations on resources.  They may automatically handle the passing in of arguments set from identifiers and some attributes.', intro_link='actions_intro')
    for action_name in sorted(resource_actions):
        action_section = section.add_new_section(action_name)
        if action_name in ['load', 'reload'] and self._resource_model.load:
            document_load_reload_action(section=action_section, action_name=action_name, resource_name=self._resource_name, event_emitter=self._resource.meta.client.meta.events, load_model=self._resource_model.load, service_model=self._service_model)
        elif action_name in modeled_actions:
            document_action(section=action_section, resource_name=self._resource_name, event_emitter=self._resource.meta.client.meta.events, action_model=modeled_actions[action_name], service_model=self._service_model)
        else:
            document_custom_method(action_section, action_name, resource_actions[action_name])