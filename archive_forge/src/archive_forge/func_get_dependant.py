import inspect
from contextlib import AsyncExitStack, contextmanager
from copy import deepcopy
from typing import (
import anyio
from fastapi import params
from fastapi._compat import (
from fastapi.background import BackgroundTasks
from fastapi.concurrency import (
from fastapi.dependencies.models import Dependant, SecurityRequirement
from fastapi.logger import logger
from fastapi.security.base import SecurityBase
from fastapi.security.oauth2 import OAuth2, SecurityScopes
from fastapi.security.open_id_connect_url import OpenIdConnect
from fastapi.utils import create_response_field, get_path_param_names
from pydantic.fields import FieldInfo
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, QueryParams, UploadFile
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response
from starlette.websockets import WebSocket
from typing_extensions import Annotated, get_args, get_origin
def get_dependant(*, path: str, call: Callable[..., Any], name: Optional[str]=None, security_scopes: Optional[List[str]]=None, use_cache: bool=True) -> Dependant:
    path_param_names = get_path_param_names(path)
    endpoint_signature = get_typed_signature(call)
    signature_params = endpoint_signature.parameters
    dependant = Dependant(call=call, name=name, path=path, security_scopes=security_scopes, use_cache=use_cache)
    for param_name, param in signature_params.items():
        is_path_param = param_name in path_param_names
        type_annotation, depends, param_field = analyze_param(param_name=param_name, annotation=param.annotation, value=param.default, is_path_param=is_path_param)
        if depends is not None:
            sub_dependant = get_param_sub_dependant(param_name=param_name, depends=depends, path=path, security_scopes=security_scopes)
            dependant.dependencies.append(sub_dependant)
            continue
        if add_non_field_param_to_dependency(param_name=param_name, type_annotation=type_annotation, dependant=dependant):
            assert param_field is None, f'Cannot specify multiple FastAPI annotations for {param_name!r}'
            continue
        assert param_field is not None
        if is_body_param(param_field=param_field, is_path_param=is_path_param):
            dependant.body_params.append(param_field)
        else:
            add_param_to_fields(field=param_field, dependant=dependant)
    return dependant