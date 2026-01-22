from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def create_openapi_schema_by_role_function(module_name: str, roles: List[OpenAPIRoleSpec], module_domains: Optional[List[str]]=None, domain_name: Optional[str]=None, domain_name_force_https: Optional[bool]=None, default_exclude_paths: Optional[List[str]]=None, replace_patches: Optional[List[Tuple[str, Union[Callable, Optional[str]]]]]=None, replace_key_start: Optional[str]=KEY_START, replace_key_end: Optional[str]=KEY_END, replace_sep_char: Optional[str]=KEY_SEP, replace_domain_key: Optional[str]=DOMAIN_KEY, verbose: Optional[bool]=True, **kwargs) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create an openapi schema by role
    """
    setup_new_module_schema(module_name=module_name, roles=roles)
    if not default_exclude_paths:
        default_exclude_paths = []

    def patch_openapi_description(role_spec: OpenAPIRoleSpec, schema: Dict[str, Union[Dict[str, Union[Dict[str, Any], Any]], Any]], request: Optional['Request']=None, force_https: Optional[bool]=None) -> Dict[str, Any]:
        """
        Patch the openapi schema description
        """
        nonlocal domain_name
        if domain_name is None:
            domain_name = get_server_domain(request=request, module_name=module_name, module_domains=module_domains, verbose=verbose, force_https=domain_name_force_https or force_https)
        if domain_name:
            replace_domain_start = f'<<{replace_domain_key}>>'
            replace_domain_end = f'>>{replace_domain_key}<<'
            schema['info']['description'] = schema['info']['description'].replace(replace_domain_start, domain_name).replace(replace_domain_end, domain_name)
            if role_spec.has_description_callable:
                schema['info']['description'] = role_spec.description_callable(schema['info']['description'], domain_name, role_spec)
        return schema

    def patch_openapi_paths(role_spec: OpenAPIRoleSpec, schema: Dict[str, Union[Dict[str, Union[Dict[str, Any], Any]], Any]]) -> Dict[str, Any]:
        """
        Patch the openapi schema paths
        """
        for path, methods in schema['paths'].items():
            for method, spec in methods.items():
                if 'tags' in spec and any((tag in role_spec.excluded_tags for tag in spec['tags'])):
                    role_spec.excluded_paths.append(path)
        for path in role_spec.excluded_paths:
            if isinstance(path, str):
                schema['paths'].pop(path, None)
            elif isinstance(path, dict):
                schema['paths'][path['path']].pop(path['method'], None)
        for path in default_exclude_paths:
            if path in role_spec.included_paths:
                continue
            if isinstance(path, str):
                schema['paths'].pop(path, None)
            elif isinstance(path, dict):
                schema['paths'][path['path']].pop(path['method'], None)
        if 'components' not in schema:
            return schema
        if 'schemas' not in schema['components']:
            return schema
        _schemas_to_remove = []
        for schema_name in role_spec.excluded_schemas:
            if '*' in schema_name:
                _schemas_to_remove += [schema for schema in schema['components']['schemas'].keys() if re.match(schema_name, schema)]
            else:
                _schemas_to_remove.append(schema_name)
        _schemas_to_remove = list(set(_schemas_to_remove))
        for schema_name in _schemas_to_remove:
            schema['components']['schemas'].pop(schema_name, None)
        if role_spec.extra_schemas:
            role_spec.populate_extra_schemas()
            schema['components']['schemas'].update(role_spec.extra_schemas_data)
        schema['components']['schemas'] = dict(sorted(schema['components']['schemas'].items(), key=lambda x: operator.itemgetter('title')(x[1])))
        return schema

    def get_openapi_schema_by_role(user_role: Optional[Union['UserRole', str]]=None, request: Optional['Request']=None, app: Optional['FastAPI']=None, force_https: Optional[bool]=None) -> Dict[str, Any]:
        """
        Get the openapi schema by role
        """
        module_schemas = get_module_by_role_schema(module_name=module_name)
        role = user_role or UserRole.ANON
        role_spec = module_schemas[role]
        if not role_spec.openapi_schema:
            if verbose:
                logger.info('Generating OpenAPI Schema', prefix=role)
            schema = copy.deepcopy(app.openapi()) if app else copy.deepcopy(get_openapi_schema(module_name))
            patch_openapi_description(role_spec=role_spec, schema=schema, request=request, force_https=domain_name_force_https or force_https)
            patch_openapi_paths(role_spec=role_spec, schema=schema)
            if replace_patches:
                for patch in replace_patches:
                    key, value = patch
                    if callable(value):
                        value = value()
                    schema = json_replace(schema=schema, key=key, value=value, key_start=replace_key_start, key_end=replace_key_end, sep_char=replace_sep_char)
            role_spec.openapi_schema = schema
            set_module_role_spec(module_name=module_name, role_spec=role_spec)
        return role_spec.openapi_schema
    return get_openapi_schema_by_role