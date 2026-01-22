import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___setuptools_pypa_io_en_latest_references_keywords_html__definitions_attr_directive(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'title': "'attr:' directive", '$id': '#/definitions/attr-directive', '$$description': ['Value is read from a module attribute. Supports callables and iterables;', 'unsupported types are cast via ``str()``'], 'type': 'object', 'additionalProperties': False, 'properties': {'attr': {'type': 'string'}}, 'required': ['attr']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all((prop in data for prop in ['attr'])):
            raise JsonSchemaValueException('' + (name_prefix or 'data') + " must contain ['attr'] properties", value=data, name='' + (name_prefix or 'data') + '', definition={'title': "'attr:' directive", '$id': '#/definitions/attr-directive', '$$description': ['Value is read from a module attribute. Supports callables and iterables;', 'unsupported types are cast via ``str()``'], 'type': 'object', 'additionalProperties': False, 'properties': {'attr': {'type': 'string'}}, 'required': ['attr']}, rule='required')
        data_keys = set(data.keys())
        if 'attr' in data_keys:
            data_keys.remove('attr')
            data__attr = data['attr']
            if not isinstance(data__attr, str):
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.attr must be string', value=data__attr, name='' + (name_prefix or 'data') + '.attr', definition={'type': 'string'}, rule='type')
        if data_keys:
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must not contain ' + str(data_keys) + ' properties', value=data, name='' + (name_prefix or 'data') + '', definition={'title': "'attr:' directive", '$id': '#/definitions/attr-directive', '$$description': ['Value is read from a module attribute. Supports callables and iterables;', 'unsupported types are cast via ``str()``'], 'type': 'object', 'additionalProperties': False, 'properties': {'attr': {'type': 'string'}}, 'required': ['attr']}, rule='additionalProperties')
    return data