import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def get_default_host() -> str:
    """Get the default Databricks workspace hostname.
    Raises an error if the hostname cannot be automatically determined.
    """
    host = os.getenv('DATABRICKS_HOST')
    if not host:
        try:
            host = get_repl_context().browserHostName
            if not host:
                raise ValueError("context doesn't contain browserHostName.")
        except Exception as e:
            raise ValueError(f"host was not set and cannot be automatically inferred. Set environment variable 'DATABRICKS_HOST'. Received error: {e}")
    host = host.lstrip('https://').lstrip('http://').rstrip('/')
    return host