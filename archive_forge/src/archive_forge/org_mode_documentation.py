from pathlib import Path
from typing import Any, List, Union
from langchain_community.document_loaders.unstructured import (


        Args:
            file_path: The path to the file to load.
            mode: The mode to load the file from. Default is "single".
            **unstructured_kwargs: Any additional keyword arguments to pass
                to the unstructured.
        