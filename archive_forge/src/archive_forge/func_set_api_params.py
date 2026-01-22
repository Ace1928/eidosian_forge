from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def set_api_params(module, module_params):
    """
    Sets module parameters to those expected by the boto3 API.
    :param module:
    :param module_params:
    :return:
    """
    api_params = dict(((k, v) for k, v in dict(module.params).items() if k in module_params and v is not None))
    return snake_dict_to_camel_dict(api_params)