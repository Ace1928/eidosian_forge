import http.client
import inspect
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
from fastapi import routing
from fastapi._compat import (
from fastapi.datastructures import DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant, get_flat_params
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.constants import METHODS_WITH_BODY, REF_PREFIX, REF_TEMPLATE
from fastapi.openapi.models import OpenAPI
from fastapi.params import Body, Param
from fastapi.responses import Response
from fastapi.types import ModelNameMap
from fastapi.utils import (
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from typing_extensions import Literal
def get_openapi_operation_metadata(*, route: routing.APIRoute, method: str, operation_ids: Set[str]) -> Dict[str, Any]:
    operation: Dict[str, Any] = {}
    if route.tags:
        operation['tags'] = route.tags
    operation['summary'] = generate_operation_summary(route=route, method=method)
    if route.description:
        operation['description'] = route.description
    operation_id = route.operation_id or route.unique_id
    if operation_id in operation_ids:
        message = f'Duplicate Operation ID {operation_id} for function ' + f'{route.endpoint.__name__}'
        file_name = getattr(route.endpoint, '__globals__', {}).get('__file__')
        if file_name:
            message += f' at {file_name}'
        warnings.warn(message, stacklevel=1)
    operation_ids.add(operation_id)
    operation['operationId'] = operation_id
    if route.deprecated:
        operation['deprecated'] = route.deprecated
    return operation