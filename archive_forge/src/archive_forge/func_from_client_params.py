from __future__ import annotations
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from langchain_core.utils import get_from_env
@classmethod
def from_client_params(cls, session_id: str, cache_name: str, ttl: timedelta, *, configuration: Optional[momento.config.Configuration]=None, api_key: Optional[str]=None, auth_token: Optional[str]=None, **kwargs: Any) -> MomentoChatMessageHistory:
    """Construct cache from CacheClient parameters."""
    try:
        from momento import CacheClient, Configurations, CredentialProvider
    except ImportError:
        raise ImportError('Could not import momento python package. Please install it with `pip install momento`.')
    if configuration is None:
        configuration = Configurations.Laptop.v1()
    try:
        api_key = auth_token or get_from_env('auth_token', 'MOMENTO_AUTH_TOKEN')
    except ValueError:
        api_key = api_key or get_from_env('api_key', 'MOMENTO_API_KEY')
    credentials = CredentialProvider.from_string(api_key)
    cache_client = CacheClient(configuration, credentials, default_ttl=ttl)
    return cls(session_id, cache_client, cache_name, ttl=ttl, **kwargs)