import warnings
from typing import TYPE_CHECKING, Optional
def set_verbose(value: bool) -> None:
    """Set a new value for the `verbose` global setting."""
    import langchain
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Importing verbose from langchain root module is no longer supported')
        langchain.verbose = value
    global _verbose
    _verbose = value