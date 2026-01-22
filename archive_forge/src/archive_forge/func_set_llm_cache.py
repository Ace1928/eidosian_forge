import warnings
from typing import TYPE_CHECKING, Optional
def set_llm_cache(value: Optional['BaseCache']) -> None:
    """Set a new LLM cache, overwriting the previous value, if any."""
    import langchain
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Importing llm_cache from langchain root module is no longer supported')
        langchain.llm_cache = value
    global _llm_cache
    _llm_cache = value