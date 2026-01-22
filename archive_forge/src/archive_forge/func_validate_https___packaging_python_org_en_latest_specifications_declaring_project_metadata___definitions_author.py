import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___packaging_python_org_en_latest_specifications_declaring_project_metadata___definitions_author(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/author', 'title': 'Author or Maintainer', '$comment': 'https://peps.python.org/pep-0621/#authors-maintainers', 'type': 'object', 'additionalProperties': False, 'properties': {'name': {'type': 'string', '$$description': ['MUST be a valid email name, i.e. whatever can be put as a name, before an', 'email, in :rfc:`822`.']}, 'email': {'type': 'string', 'format': 'idn-email', 'description': 'MUST be a valid email address'}}}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_keys = set(data.keys())
        if 'name' in data_keys:
            data_keys.remove('name')
            data__name = data['name']
            if not isinstance(data__name, str):
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.name must be string', value=data__name, name='' + (name_prefix or 'data') + '.name', definition={'type': 'string', '$$description': ['MUST be a valid email name, i.e. whatever can be put as a name, before an', 'email, in :rfc:`822`.']}, rule='type')
        if 'email' in data_keys:
            data_keys.remove('email')
            data__email = data['email']
            if not isinstance(data__email, str):
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.email must be string', value=data__email, name='' + (name_prefix or 'data') + '.email', definition={'type': 'string', 'format': 'idn-email', 'description': 'MUST be a valid email address'}, rule='type')
            if isinstance(data__email, str):
                if not REGEX_PATTERNS['idn-email_re_pattern'].match(data__email):
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + '.email must be idn-email', value=data__email, name='' + (name_prefix or 'data') + '.email', definition={'type': 'string', 'format': 'idn-email', 'description': 'MUST be a valid email address'}, rule='format')
        if data_keys:
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must not contain ' + str(data_keys) + ' properties', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/author', 'title': 'Author or Maintainer', '$comment': 'https://peps.python.org/pep-0621/#authors-maintainers', 'type': 'object', 'additionalProperties': False, 'properties': {'name': {'type': 'string', '$$description': ['MUST be a valid email name, i.e. whatever can be put as a name, before an', 'email, in :rfc:`822`.']}, 'email': {'type': 'string', 'format': 'idn-email', 'description': 'MUST be a valid email address'}}}, rule='additionalProperties')
    return data