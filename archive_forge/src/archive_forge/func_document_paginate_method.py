import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
def document_paginate_method(section, paginator_name, event_emitter, service_model, paginator_config, include_signature=True):
    """Documents the paginate method of a paginator

    :param section: The section to write to

    :param paginator_name: The name of the paginator. It is snake cased.

    :param event_emitter: The event emitter to use to emit events

    :param service_model: The service model

    :param paginator_config: The paginator config associated to a particular
        paginator.

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    operation_model = service_model.operation_model(paginator_name)
    pagination_config_members = OrderedDict()
    pagination_config_members['MaxItems'] = DocumentedShape(name='MaxItems', type_name='integer', documentation='<p>The total number of items to return. If the total number of items available is more than the value specified in max-items then a <code>NextToken</code> will be provided in the output that you can use to resume pagination.</p>')
    if paginator_config.get('limit_key', None):
        pagination_config_members['PageSize'] = DocumentedShape(name='PageSize', type_name='integer', documentation='<p>The size of each page.<p>')
    pagination_config_members['StartingToken'] = DocumentedShape(name='StartingToken', type_name='string', documentation='<p>A token to specify where to start paginating. This is the <code>NextToken</code> from a previous response.</p>')
    botocore_pagination_params = [DocumentedShape(name='PaginationConfig', type_name='structure', documentation='<p>A dictionary that provides parameters to control pagination.</p>', members=pagination_config_members)]
    botocore_pagination_response_params = [DocumentedShape(name='NextToken', type_name='string', documentation='<p>A token to resume pagination.</p>')]
    service_pagination_params = []
    if isinstance(paginator_config['input_token'], list):
        service_pagination_params += paginator_config['input_token']
    else:
        service_pagination_params.append(paginator_config['input_token'])
    if paginator_config.get('limit_key', None):
        service_pagination_params.append(paginator_config['limit_key'])
    service_pagination_response_params = []
    if isinstance(paginator_config['output_token'], list):
        service_pagination_response_params += paginator_config['output_token']
    else:
        service_pagination_response_params.append(paginator_config['output_token'])
    paginate_description = 'Creates an iterator that will paginate through responses from :py:meth:`{}.Client.{}`.'.format(get_service_module_name(service_model), xform_name(paginator_name))
    document_model_driven_method(section, 'paginate', operation_model, event_emitter=event_emitter, method_description=paginate_description, example_prefix='response_iterator = paginator.paginate', include_input=botocore_pagination_params, include_output=botocore_pagination_response_params, exclude_input=service_pagination_params, exclude_output=service_pagination_response_params, include_signature=include_signature)