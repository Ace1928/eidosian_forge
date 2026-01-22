import base64
import functools
import inspect
import json
import logging
import posixpath
import re
import textwrap
import warnings
from typing import Any, AsyncGenerator, List, Optional
from urllib.parse import urlparse
from starlette.responses import StreamingResponse
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
from mlflow.utils.uri import append_to_uri_path
def check_configuration_deprecated_fields(config):
    if 'routes' in config:
        warnings.warn("The 'routes' configuration key has been deprecated and will be removed in an upcoming release. Use 'endpoints' instead.", FutureWarning, stacklevel=2)
    routes = config.get('routes', []) or config.get('endpoints', [])
    for route in routes:
        if 'route_type' in route:
            warnings.warn("The 'route_type' configuration key has been deprecated and will be removed in an upcoming release. Use 'endpoint_type' instead.", FutureWarning, stacklevel=2)
            break