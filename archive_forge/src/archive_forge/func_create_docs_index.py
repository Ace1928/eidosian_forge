from __future__ import annotations
import json
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from fastapi import Request
from fastapi.background import BackgroundTasks
from ..utils.lazy import get_az_settings, get_az_mtg_api, get_az_resource_schema, logger
from ..utils.helpers import get_hashed_key, create_code_challenge, parse_scopes, encode_params_to_url
from typing import Optional, List, Dict, Any, Union, Type
def create_docs_index(self, schema: Dict[str, Any]):
    """
        Creates the docs index
        """
    for path in schema.get('paths', {}):
        for method in schema['paths'][path]:
            if 'operationId' in schema['paths'][path][method]:
                doc_name = schema['paths'][path][method]['summary'].replace(' ', '').lower()
                self.docs_schema_index[doc_name] = schema['paths'][path][method]['operationId']