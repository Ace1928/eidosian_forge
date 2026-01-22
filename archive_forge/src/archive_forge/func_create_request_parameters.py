import re
import jmespath
from botocore import xform_name
from ..exceptions import ResourceLoadException
def create_request_parameters(parent, request_model, params=None, index=None):
    """
    Handle request parameters that can be filled in from identifiers,
    resource data members or constants.

    By passing ``params``, you can invoke this method multiple times and
    build up a parameter dict over time, which is particularly useful
    for reverse JMESPath expressions that append to lists.

    :type parent: ServiceResource
    :param parent: The resource instance to which this action is attached.
    :type request_model: :py:class:`~boto3.resources.model.Request`
    :param request_model: The action request model.
    :type params: dict
    :param params: If set, then add to this existing dict. It is both
                   edited in-place and returned.
    :type index: int
    :param index: The position of an item within a list
    :rtype: dict
    :return: Pre-filled parameters to be sent to the request operation.
    """
    if params is None:
        params = {}
    for param in request_model.params:
        source = param.source
        target = param.target
        if source == 'identifier':
            value = getattr(parent, xform_name(param.name))
        elif source == 'data':
            value = get_data_member(parent, param.path)
        elif source in ['string', 'integer', 'boolean']:
            value = param.value
        elif source == 'input':
            continue
        else:
            raise NotImplementedError('Unsupported source type: {0}'.format(source))
        build_param_structure(params, target, value, index)
    return params