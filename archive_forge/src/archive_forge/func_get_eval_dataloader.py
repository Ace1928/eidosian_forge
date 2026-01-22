import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Type
from torch.utils.data import DataLoader, Dataset, IterableDataset
import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.util import PublicAPI
def get_eval_dataloader(self, eval_dataset: Optional[Dataset]=None) -> DataLoader:
    if eval_dataset is None:
        eval_dataset = self.eval_dataset
    if isinstance(eval_dataset, _IterableFromIterator):
        dataset = RayTorchIterableDataset(eval_dataset)
        return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    else:
        return super().get_eval_dataloader(eval_dataset)