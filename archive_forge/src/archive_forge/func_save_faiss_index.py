import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def save_faiss_index(self, index_name: str, file: Union[str, PurePath], storage_options: Optional[Dict]=None):
    """Save a FaissIndex on disk.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
    index = self.get_index(index_name)
    if not isinstance(index, FaissIndex):
        raise ValueError(f"Index '{index_name}' is not a FaissIndex but a '{type(index)}'")
    index.save(file, storage_options=storage_options)
    logger.info(f'Saved FaissIndex {index_name} at {file}')