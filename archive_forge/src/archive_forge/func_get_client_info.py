from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
def get_client_info(module: Optional[str]=None) -> 'ClientInfo':
    """Return a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    try:
        from google.api_core.gapic_v1.client_info import ClientInfo
    except ImportError as exc:
        raise ImportError('Could not import ClientInfo. Please, install it with pip install google-api-core') from exc
    langchain_version = metadata.version('langchain')
    client_library_version = f'{langchain_version}-{module}' if module else langchain_version
    return ClientInfo(client_library_version=client_library_version, user_agent=f'langchain/{client_library_version}')