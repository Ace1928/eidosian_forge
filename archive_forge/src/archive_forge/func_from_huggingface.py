import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@PublicAPI
def from_huggingface(dataset: Union['datasets.Dataset', 'datasets.IterableDataset']) -> Union[MaterializedDataset, Dataset]:
    """Create a :class:`~ray.data.MaterializedDataset` from a
    `Hugging Face Datasets Dataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset/>`_
    or a :class:`~ray.data.Dataset` from a `Hugging Face Datasets IterableDataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.IterableDataset/>`_.
    For an `IterableDataset`, we use a streaming implementation to read data.

    Example:

        ..
            The following `testoutput` is mocked to avoid illustrating download
            logs like "Downloading and preparing dataset 162.17 MiB".

        .. testcode::

            import ray
            import datasets

            hf_dataset = datasets.load_dataset("tweet_eval", "emotion")
            ray_ds = ray.data.from_huggingface(hf_dataset["train"])
            print(ray_ds)

            hf_dataset_stream = datasets.load_dataset("tweet_eval", "emotion", streaming=True)
            ray_ds_stream = ray.data.from_huggingface(hf_dataset_stream["train"])
            print(ray_ds_stream)

        .. testoutput::
            :options: +MOCK

            MaterializedDataset(
                num_blocks=...,
                num_rows=3257,
                schema={text: string, label: int64}
            )
            Dataset(
                num_blocks=...,
                num_rows=3257,
                schema={text: string, label: int64}
            )

    Args:
        dataset: A `Hugging Face Datasets Dataset`_ or `Hugging Face Datasets IterableDataset`_.
            `DatasetDict <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict/>`_
            and `IterableDatasetDict <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.IterableDatasetDict/>`_
            are not supported.

    Returns:
        A :class:`~ray.data.Dataset` holding rows from the `Hugging Face Datasets Dataset`_.
    """
    import datasets
    if isinstance(dataset, datasets.IterableDataset):
        from ray.data.datasource.huggingface_datasource import HuggingFaceDatasource
        return read_datasource(HuggingFaceDatasource(dataset=dataset))
    if isinstance(dataset, datasets.Dataset):
        hf_ds_arrow = dataset.with_format('arrow')
        ray_ds = from_arrow(hf_ds_arrow[:])
        return ray_ds
    elif isinstance(dataset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
        available_keys = list(dataset.keys())
        raise DeprecationWarning(f"You provided a Hugging Face DatasetDict or IterableDatasetDict, which contains multiple datasets, but `from_huggingface` now only accepts a single Hugging Face Dataset. To convert just a single Hugging Face Dataset to a Ray Dataset, specify a split. For example, `ray.data.from_huggingface(my_dataset_dictionary['{available_keys[0]}'])`. Available splits are {available_keys}.")
    else:
        raise TypeError(f'`dataset` must be a `datasets.Dataset`, but got {type(dataset)}')