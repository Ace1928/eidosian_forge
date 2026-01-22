import os
from typing import BinaryIO, Optional, Union
import numpy as np
import pyarrow.parquet as pq
from .. import Audio, Dataset, Features, Image, NamedSplit, Value, config
from ..features.features import FeatureType, _visit
from ..formatting import query_table
from ..packaged_modules import _PACKAGED_DATASETS_MODULES
from ..packaged_modules.parquet.parquet import Parquet
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader
def get_writer_batch_size(features: Features) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files.
    The default in `datasets` is 1,000 but we lower it to 100 for image datasets.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    This can be improved to get optimized size for querying/iterating
    but at least it matches the dataset viewer expectations on HF.

    Args:
        ds_config_info (`datasets.info.DatasetInfo`):
            Dataset info from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
            If `None`, then it will use the `datasets` default.
    """
    batch_size = np.inf

    def set_batch_size(feature: FeatureType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image):
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
        elif isinstance(feature, Audio):
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS)
        elif isinstance(feature, Value) and feature.dtype == 'binary':
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)
    _visit(features, set_batch_size)
    return None if batch_size is np.inf else batch_size