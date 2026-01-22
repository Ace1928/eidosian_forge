from typing import List, Optional, TypeVar
from .arrow_dataset import Dataset, _concatenate_map_style_datasets, _interleave_map_style_datasets
from .dataset_dict import DatasetDict, IterableDatasetDict
from .info import DatasetInfo
from .iterable_dataset import IterableDataset, _concatenate_iterable_datasets, _interleave_iterable_datasets
from .splits import NamedSplit
from .utils import logging
from .utils.py_utils import Literal
def concatenate_datasets(dsets: List[DatasetType], info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, axis: int=0) -> DatasetType:
    """
    Converts a list of [`Dataset`] with the same schema into a single [`Dataset`].

    Args:
        dsets (`List[datasets.Dataset]`):
            List of Datasets to concatenate.
        info (`DatasetInfo`, *optional*):
            Dataset information, like description, citation, etc.
        split (`NamedSplit`, *optional*):
            Name of the dataset split.
        axis (`{0, 1}`, defaults to `0`):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).

            <Added version="1.6.0"/>

    Example:

    ```py
    >>> ds3 = concatenate_datasets([ds1, ds2])
    ```
    """
    if not dsets:
        raise ValueError('Unable to concatenate an empty list of datasets.')
    for i, dataset in enumerate(dsets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(f'Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is an empty dataset dictionary.')
                raise ValueError(f"Dataset at position {i} has at least one split: {list(dataset)}\nPlease pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']")
            raise ValueError(f'Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}.')
        if i == 0:
            dataset_type, other_type = (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
        elif not isinstance(dataset, dataset_type):
            raise ValueError(f'Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects.')
    if dataset_type is Dataset:
        return _concatenate_map_style_datasets(dsets, info=info, split=split, axis=axis)
    else:
        return _concatenate_iterable_datasets(dsets, info=info, split=split, axis=axis)