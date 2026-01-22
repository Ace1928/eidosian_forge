import asyncio
import dataclasses
import email.message
import inspect
import json
from contextlib import AsyncExitStack
from enum import Enum, IntEnum
from typing import (
from fastapi import params
from fastapi._compat import (
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import (
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import (
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import (
from pydantic import BaseModel
from starlette import routing
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import (
from starlette.routing import Mount as Mount  # noqa
from starlette.types import ASGIApp, Lifespan, Scope
from starlette.websockets import WebSocket
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def include_router(self, router: Annotated['APIRouter', Doc('The `APIRouter` to include.')], *, prefix: Annotated[str, Doc('An optional path prefix for the router.')]='', tags: Annotated[Optional[List[Union[str, Enum]]], Doc('\n                A list of tags to be applied to all the *path operations* in this\n                router.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]=None, dependencies: Annotated[Optional[Sequence[params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to all the\n                *path operations* in this router.\n\n                Read more about it in the\n                [FastAPI docs for Bigger Applications - Multiple Files](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n                ')]=None, default_response_class: Annotated[Type[Response], Doc('\n                The default response class to be used.\n\n                Read more in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#default-response-class).\n                ')]=Default(JSONResponse), responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses to be shown in OpenAPI.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Additional Responses in OpenAPI](https://fastapi.tiangolo.com/advanced/additional-responses/).\n\n                And in the\n                [FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n                ')]=None, callbacks: Annotated[Optional[List[BaseRoute]], Doc('\n                OpenAPI callbacks that should apply to all *path operations* in this\n                router.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ')]=None, deprecated: Annotated[Optional[bool], Doc('\n                Mark all *path operations* in this router as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]=None, include_in_schema: Annotated[bool, Doc('\n                Include (or not) all the *path operations* in this router in the\n                generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n                ')]=True, generate_unique_id_function: Annotated[Callable[[APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]=Default(generate_unique_id)) -> None:
    """
        Include another `APIRouter` in the same current `APIRouter`.

        Read more about it in the
        [FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/).

        ## Example

        ```python
        from fastapi import APIRouter, FastAPI

        app = FastAPI()
        internal_router = APIRouter()
        users_router = APIRouter()

        @users_router.get("/users/")
        def read_users():
            return [{"name": "Rick"}, {"name": "Morty"}]

        internal_router.include_router(users_router)
        app.include_router(internal_router)
        ```
        """
    if prefix:
        assert prefix.startswith('/'), "A path prefix must start with '/'"
        assert not prefix.endswith('/'), "A path prefix must not end with '/', as the routes will start with '/'"
    else:
        for r in router.routes:
            path = getattr(r, 'path')
            name = getattr(r, 'name', 'unknown')
            if path is not None and (not path):
                raise FastAPIError(f'Prefix and path cannot be both empty (path operation: {name})')
    if responses is None:
        responses = {}
    for route in router.routes:
        if isinstance(route, APIRoute):
            combined_responses = {**responses, **route.responses}
            use_response_class = get_value_or_default(route.response_class, router.default_response_class, default_response_class, self.default_response_class)
            current_tags = []
            if tags:
                current_tags.extend(tags)
            if route.tags:
                current_tags.extend(route.tags)
            current_dependencies: List[params.Depends] = []
            if dependencies:
                current_dependencies.extend(dependencies)
            if route.dependencies:
                current_dependencies.extend(route.dependencies)
            current_callbacks = []
            if callbacks:
                current_callbacks.extend(callbacks)
            if route.callbacks:
                current_callbacks.extend(route.callbacks)
            current_generate_unique_id = get_value_or_default(route.generate_unique_id_function, router.generate_unique_id_function, generate_unique_id_function, self.generate_unique_id_function)
            self.add_api_route(prefix + route.path, route.endpoint, response_model=route.response_model, status_code=route.status_code, tags=current_tags, dependencies=current_dependencies, summary=route.summary, description=route.description, response_description=route.response_description, responses=combined_responses, deprecated=route.deprecated or deprecated or self.deprecated, methods=route.methods, operation_id=route.operation_id, response_model_include=route.response_model_include, response_model_exclude=route.response_model_exclude, response_model_by_alias=route.response_model_by_alias, response_model_exclude_unset=route.response_model_exclude_unset, response_model_exclude_defaults=route.response_model_exclude_defaults, response_model_exclude_none=route.response_model_exclude_none, include_in_schema=route.include_in_schema and self.include_in_schema and include_in_schema, response_class=use_response_class, name=route.name, route_class_override=type(route), callbacks=current_callbacks, openapi_extra=route.openapi_extra, generate_unique_id_function=current_generate_unique_id)
        elif isinstance(route, routing.Route):
            methods = list(route.methods or [])
            self.add_route(prefix + route.path, route.endpoint, methods=methods, include_in_schema=route.include_in_schema, name=route.name)
        elif isinstance(route, APIWebSocketRoute):
            current_dependencies = []
            if dependencies:
                current_dependencies.extend(dependencies)
            if route.dependencies:
                current_dependencies.extend(route.dependencies)
            self.add_api_websocket_route(prefix + route.path, route.endpoint, dependencies=current_dependencies, name=route.name)
        elif isinstance(route, routing.WebSocketRoute):
            self.add_websocket_route(prefix + route.path, route.endpoint, name=route.name)
    for handler in router.on_startup:
        self.add_event_handler('startup', handler)
    for handler in router.on_shutdown:
        self.add_event_handler('shutdown', handler)