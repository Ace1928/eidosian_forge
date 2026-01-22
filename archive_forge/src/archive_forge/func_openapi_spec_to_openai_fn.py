from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.openapi import OpenAPISpec
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.tools import APIOperation
def openapi_spec_to_openai_fn(spec: OpenAPISpec) -> Tuple[List[Dict[str, Any]], Callable]:
    """Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI
        functions.

    Args:
        spec: OpenAPI spec to convert.

    Returns:
        Tuple of the OpenAI functions JSON schema and a default function for executing
            a request based on the OpenAI function schema.
    """
    if not spec.paths:
        return ([], lambda: None)
    functions = []
    _name_to_call_map = {}
    for path in spec.paths:
        path_params = {(p.name, p.param_in): p for p in spec.get_parameters_for_path(path)}
        for method in spec.get_methods_for_path(path):
            request_args = {}
            op = spec.get_operation(path, method)
            op_params = path_params.copy()
            for param in spec.get_parameters_for_operation(op):
                op_params[param.name, param.param_in] = param
            params_by_type = defaultdict(list)
            for name_loc, p in op_params.items():
                params_by_type[name_loc[1]].append(p)
            param_loc_to_arg_name = {'query': 'params', 'header': 'headers', 'cookie': 'cookies', 'path': 'path_params'}
            for param_loc, arg_name in param_loc_to_arg_name.items():
                if params_by_type[param_loc]:
                    request_args[arg_name] = _openapi_params_to_json_schema(params_by_type[param_loc], spec)
            request_body = spec.get_request_body_for_operation(op)
            if request_body and request_body.content:
                media_types = {}
                for media_type, media_type_object in request_body.content.items():
                    if media_type_object.media_type_schema:
                        schema = spec.get_schema(media_type_object.media_type_schema)
                        media_types[media_type] = json.loads(schema.json(exclude_none=True))
                if len(media_types) == 1:
                    media_type, schema_dict = list(media_types.items())[0]
                    key = 'json' if media_type == 'application/json' else 'data'
                    request_args[key] = schema_dict
                elif len(media_types) > 1:
                    request_args['data'] = {'anyOf': list(media_types.values())}
            api_op = APIOperation.from_openapi_spec(spec, path, method)
            fn = {'name': api_op.operation_id, 'description': api_op.description, 'parameters': {'type': 'object', 'properties': request_args}}
            functions.append(fn)
            _name_to_call_map[fn['name']] = {'method': method, 'url': api_op.base_url + api_op.path}

    def default_call_api(name: str, fn_args: dict, headers: Optional[dict]=None, params: Optional[dict]=None, **kwargs: Any) -> Any:
        method = _name_to_call_map[name]['method']
        url = _name_to_call_map[name]['url']
        path_params = fn_args.pop('path_params', {})
        url = _format_url(url, path_params)
        if 'data' in fn_args and isinstance(fn_args['data'], dict):
            fn_args['data'] = json.dumps(fn_args['data'])
        _kwargs = {**fn_args, **kwargs}
        if headers is not None:
            if 'headers' in _kwargs:
                _kwargs['headers'].update(headers)
            else:
                _kwargs['headers'] = headers
        if params is not None:
            if 'params' in _kwargs:
                _kwargs['params'].update(params)
            else:
                _kwargs['params'] = params
        return requests.request(method, url, **_kwargs)
    return (functions, default_call_api)