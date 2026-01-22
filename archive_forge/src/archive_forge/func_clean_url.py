import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
def clean_url(url: str) -> str:
    """Remove trailing slash and /api from url if present."""
    if url.endswith('/api'):
        return url[:-4]
    elif url.endswith('/'):
        return url[:-1]
    else:
        return url