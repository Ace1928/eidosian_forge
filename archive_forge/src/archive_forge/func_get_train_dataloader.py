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
def get_train_dataloader(self) -> DataLoader:
    if isinstance(self.train_dataset, _IterableFromIterator):
        dataset = RayTorchIterableDataset(self.train_dataset)
        return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    else:
        return super().get_train_dataloader()