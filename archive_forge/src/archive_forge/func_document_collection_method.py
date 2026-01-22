from botocore import xform_name
from botocore.docs.method import get_instance_public_methods
from botocore.docs.utils import DocumentedShape
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import add_resource_type_overview
def document_collection_method(section, resource_name, action_name, event_emitter, collection_model, service_model, include_signature=True):
    """Documents a collection method

    :param section: The section to write to

    :param resource_name: The name of the resource

    :param action_name: The name of collection action. Currently only
        can be all, filter, limit, or page_size

    :param event_emitter: The event emitter to use to emit events

    :param collection_model: The model of the collection

    :param service_model: The model of the service

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    operation_model = service_model.operation_model(collection_model.request.operation)
    underlying_operation_members = []
    if operation_model.input_shape:
        underlying_operation_members = operation_model.input_shape.members
    example_resource_name = xform_name(resource_name)
    if service_model.service_name == resource_name:
        example_resource_name = resource_name
    custom_action_info_dict = {'all': {'method_description': 'Creates an iterable of all %s resources in the collection.' % collection_model.resource.type, 'example_prefix': '%s_iterator = %s.%s.all' % (xform_name(collection_model.resource.type), example_resource_name, collection_model.name), 'exclude_input': underlying_operation_members}, 'filter': {'method_description': 'Creates an iterable of all %s resources in the collection filtered by kwargs passed to method.' % collection_model.resource.type + 'A %s collection will include all resources by default if no filters are provided, and extreme caution should be taken when performing actions on all resources.' % collection_model.resource.type, 'example_prefix': '%s_iterator = %s.%s.filter' % (xform_name(collection_model.resource.type), example_resource_name, collection_model.name), 'exclude_input': get_resource_ignore_params(collection_model.request.params)}, 'limit': {'method_description': 'Creates an iterable up to a specified amount of %s resources in the collection.' % collection_model.resource.type, 'example_prefix': '%s_iterator = %s.%s.limit' % (xform_name(collection_model.resource.type), example_resource_name, collection_model.name), 'include_input': [DocumentedShape(name='count', type_name='integer', documentation='The limit to the number of resources in the iterable.')], 'exclude_input': underlying_operation_members}, 'page_size': {'method_description': 'Creates an iterable of all %s resources in the collection, but limits the number of items returned by each service call by the specified amount.' % collection_model.resource.type, 'example_prefix': '%s_iterator = %s.%s.page_size' % (xform_name(collection_model.resource.type), example_resource_name, collection_model.name), 'include_input': [DocumentedShape(name='count', type_name='integer', documentation='The number of items returned by each service call')], 'exclude_input': underlying_operation_members}}
    if action_name in custom_action_info_dict:
        action_info = custom_action_info_dict[action_name]
        document_model_driven_resource_method(section=section, method_name=action_name, operation_model=operation_model, event_emitter=event_emitter, resource_action_model=collection_model, include_signature=include_signature, **action_info)