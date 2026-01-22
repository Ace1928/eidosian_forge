import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
@root_validator
def validate_local_or_remote(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that either local file paths are given, or remote API docset ID.

        Args:
            values: The values to validate.

        Returns:
            The validated values.
        """
    if values.get('file_paths') and values.get('docset_id'):
        raise ValueError('Cannot specify both file_paths and remote API docset_id')
    if not values.get('file_paths') and (not values.get('docset_id')):
        raise ValueError('Must specify either file_paths or remote API docset_id')
    if values.get('docset_id') and (not values.get('access_token')):
        raise ValueError('Must specify access token if using remote API docset_id')
    return values