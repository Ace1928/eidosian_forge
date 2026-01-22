import json
import logging
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
def get_azure_sdk_function(client: Any, function_name: str) -> Callable:
    """Retrieve a callable function from Azure SDK client object.

    Newer versions of the various client SDKs renamed function names to
    have a begin_ prefix. This function supports both the old and new
    versions of the SDK by first trying the old name and falling back to
    the prefixed new name.
    """
    func = getattr(client, function_name, getattr(client, f'begin_{function_name}', None))
    if func is None:
        raise AttributeError("'{obj}' object has no {func} or begin_{func} attribute".format(obj={client.__name__}, func=function_name))
    return func