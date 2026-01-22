from importlib import metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
def init_vertexai(project: Optional[str]=None, location: Optional[str]=None, credentials: Optional['Credentials']=None) -> None:
    """Init Vertex AI.

    Args:
        project: The default GCP project to use when making Vertex API calls.
        location: The default location to use when making API calls.
        credentials: The default custom
            credentials to use when making API calls. If not provided credentials
            will be ascertained from the environment.

    Raises:
        ImportError: If importing vertexai SDK did not succeed.
    """
    try:
        import vertexai
    except ImportError:
        raise_vertex_import_error()
    vertexai.init(project=project, location=location, credentials=credentials)