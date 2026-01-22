from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types
def map_complex_type(complex_type, type_map):
    """
    Allows to cast elements within a dictionary to a specific type
    Example of usage:

    DEPLOYMENT_CONFIGURATION_TYPE_MAP = {
        'maximum_percent': 'int',
        'minimum_healthy_percent': 'int'
    }

    deployment_configuration = map_complex_type(module.params['deployment_configuration'],
                                                DEPLOYMENT_CONFIGURATION_TYPE_MAP)

    This ensures all keys within the root element are casted and valid integers
    """
    if complex_type is None:
        return
    new_type = type(complex_type)()
    if isinstance(complex_type, dict):
        for key in complex_type:
            if key in type_map:
                if isinstance(type_map[key], list):
                    new_type[key] = map_complex_type(complex_type[key], type_map[key][0])
                else:
                    new_type[key] = map_complex_type(complex_type[key], type_map[key])
            else:
                new_type[key] = complex_type[key]
    elif isinstance(complex_type, list):
        for i in range(len(complex_type)):
            new_type.append(map_complex_type(complex_type[i], type_map))
    elif type_map:
        return globals()['__builtins__'][type_map](complex_type)
    return new_type