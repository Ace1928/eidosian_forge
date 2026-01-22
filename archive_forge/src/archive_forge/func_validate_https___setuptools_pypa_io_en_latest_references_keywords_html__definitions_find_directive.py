import re
from .fastjsonschema_exceptions import JsonSchemaValueException
def validate_https___setuptools_pypa_io_en_latest_references_keywords_html__definitions_find_directive(data, custom_formats={}, name_prefix=None):
    if not isinstance(data, dict):
        raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must be object', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/find-directive', 'title': "'find:' directive", 'type': 'object', 'additionalProperties': False, 'properties': {'find': {'type': 'object', '$$description': ['Dynamic `package discovery', '<https://setuptools.pypa.io/en/latest/userguide/package_discovery.html>`_.'], 'additionalProperties': False, 'properties': {'where': {'description': 'Directories to be searched for packages (Unix-style relative path)', 'type': 'array', 'items': {'type': 'string'}}, 'exclude': {'type': 'array', '$$description': ['Exclude packages that match the values listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'include': {'type': 'array', '$$description': ['Restrict the found packages to just the ones listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'namespaces': {'type': 'boolean', '$$description': ['When ``True``, directories without a ``__init__.py`` file will also', 'be scanned for :pep:`420`-style implicit namespaces']}}}}}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_keys = set(data.keys())
        if 'find' in data_keys:
            data_keys.remove('find')
            data__find = data['find']
            if not isinstance(data__find, dict):
                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find must be object', value=data__find, name='' + (name_prefix or 'data') + '.find', definition={'type': 'object', '$$description': ['Dynamic `package discovery', '<https://setuptools.pypa.io/en/latest/userguide/package_discovery.html>`_.'], 'additionalProperties': False, 'properties': {'where': {'description': 'Directories to be searched for packages (Unix-style relative path)', 'type': 'array', 'items': {'type': 'string'}}, 'exclude': {'type': 'array', '$$description': ['Exclude packages that match the values listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'include': {'type': 'array', '$$description': ['Restrict the found packages to just the ones listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'namespaces': {'type': 'boolean', '$$description': ['When ``True``, directories without a ``__init__.py`` file will also', 'be scanned for :pep:`420`-style implicit namespaces']}}}, rule='type')
            data__find_is_dict = isinstance(data__find, dict)
            if data__find_is_dict:
                data__find_keys = set(data__find.keys())
                if 'where' in data__find_keys:
                    data__find_keys.remove('where')
                    data__find__where = data__find['where']
                    if not isinstance(data__find__where, (list, tuple)):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.where must be array', value=data__find__where, name='' + (name_prefix or 'data') + '.find.where', definition={'description': 'Directories to be searched for packages (Unix-style relative path)', 'type': 'array', 'items': {'type': 'string'}}, rule='type')
                    data__find__where_is_list = isinstance(data__find__where, (list, tuple))
                    if data__find__where_is_list:
                        data__find__where_len = len(data__find__where)
                        for data__find__where_x, data__find__where_item in enumerate(data__find__where):
                            if not isinstance(data__find__where_item, str):
                                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.where[{data__find__where_x}]'.format(**locals()) + ' must be string', value=data__find__where_item, name='' + (name_prefix or 'data') + '.find.where[{data__find__where_x}]'.format(**locals()) + '', definition={'type': 'string'}, rule='type')
                if 'exclude' in data__find_keys:
                    data__find_keys.remove('exclude')
                    data__find__exclude = data__find['exclude']
                    if not isinstance(data__find__exclude, (list, tuple)):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.exclude must be array', value=data__find__exclude, name='' + (name_prefix or 'data') + '.find.exclude', definition={'type': 'array', '$$description': ['Exclude packages that match the values listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, rule='type')
                    data__find__exclude_is_list = isinstance(data__find__exclude, (list, tuple))
                    if data__find__exclude_is_list:
                        data__find__exclude_len = len(data__find__exclude)
                        for data__find__exclude_x, data__find__exclude_item in enumerate(data__find__exclude):
                            if not isinstance(data__find__exclude_item, str):
                                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.exclude[{data__find__exclude_x}]'.format(**locals()) + ' must be string', value=data__find__exclude_item, name='' + (name_prefix or 'data') + '.find.exclude[{data__find__exclude_x}]'.format(**locals()) + '', definition={'type': 'string'}, rule='type')
                if 'include' in data__find_keys:
                    data__find_keys.remove('include')
                    data__find__include = data__find['include']
                    if not isinstance(data__find__include, (list, tuple)):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.include must be array', value=data__find__include, name='' + (name_prefix or 'data') + '.find.include', definition={'type': 'array', '$$description': ['Restrict the found packages to just the ones listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, rule='type')
                    data__find__include_is_list = isinstance(data__find__include, (list, tuple))
                    if data__find__include_is_list:
                        data__find__include_len = len(data__find__include)
                        for data__find__include_x, data__find__include_item in enumerate(data__find__include):
                            if not isinstance(data__find__include_item, str):
                                raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.include[{data__find__include_x}]'.format(**locals()) + ' must be string', value=data__find__include_item, name='' + (name_prefix or 'data') + '.find.include[{data__find__include_x}]'.format(**locals()) + '', definition={'type': 'string'}, rule='type')
                if 'namespaces' in data__find_keys:
                    data__find_keys.remove('namespaces')
                    data__find__namespaces = data__find['namespaces']
                    if not isinstance(data__find__namespaces, bool):
                        raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find.namespaces must be boolean', value=data__find__namespaces, name='' + (name_prefix or 'data') + '.find.namespaces', definition={'type': 'boolean', '$$description': ['When ``True``, directories without a ``__init__.py`` file will also', 'be scanned for :pep:`420`-style implicit namespaces']}, rule='type')
                if data__find_keys:
                    raise JsonSchemaValueException('' + (name_prefix or 'data') + '.find must not contain ' + str(data__find_keys) + ' properties', value=data__find, name='' + (name_prefix or 'data') + '.find', definition={'type': 'object', '$$description': ['Dynamic `package discovery', '<https://setuptools.pypa.io/en/latest/userguide/package_discovery.html>`_.'], 'additionalProperties': False, 'properties': {'where': {'description': 'Directories to be searched for packages (Unix-style relative path)', 'type': 'array', 'items': {'type': 'string'}}, 'exclude': {'type': 'array', '$$description': ['Exclude packages that match the values listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'include': {'type': 'array', '$$description': ['Restrict the found packages to just the ones listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'namespaces': {'type': 'boolean', '$$description': ['When ``True``, directories without a ``__init__.py`` file will also', 'be scanned for :pep:`420`-style implicit namespaces']}}}, rule='additionalProperties')
        if data_keys:
            raise JsonSchemaValueException('' + (name_prefix or 'data') + ' must not contain ' + str(data_keys) + ' properties', value=data, name='' + (name_prefix or 'data') + '', definition={'$id': '#/definitions/find-directive', 'title': "'find:' directive", 'type': 'object', 'additionalProperties': False, 'properties': {'find': {'type': 'object', '$$description': ['Dynamic `package discovery', '<https://setuptools.pypa.io/en/latest/userguide/package_discovery.html>`_.'], 'additionalProperties': False, 'properties': {'where': {'description': 'Directories to be searched for packages (Unix-style relative path)', 'type': 'array', 'items': {'type': 'string'}}, 'exclude': {'type': 'array', '$$description': ['Exclude packages that match the values listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'include': {'type': 'array', '$$description': ['Restrict the found packages to just the ones listed in this field.', "Can container shell-style wildcards (e.g. ``'pkg.*'``)"], 'items': {'type': 'string'}}, 'namespaces': {'type': 'boolean', '$$description': ['When ``True``, directories without a ``__init__.py`` file will also', 'be scanned for :pep:`420`-style implicit namespaces']}}}}}, rule='additionalProperties')
    return data