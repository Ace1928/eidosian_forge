from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
from ray.actor import ActorHandle
from ray.data import Dataset
def get_dataset_shards(self, training_worker_handles: List[ActorHandle]) -> List[Optional[Union[Dataset, Dict[str, Dataset]]]]:
    """Returns Dataset splits based off the spec and the given training workers

        Args:
            training_worker_handles: A list of the training worker actor handles.

        Returns:
            A list of Dataset shards or list of dictionaries of Dataset shards,
                one for each training worker.

        """
    if not self.dataset_or_dict:
        return [None] * len(training_worker_handles)
    if self.dataset_split_fn is None:
        return self._default_split_fn(training_worker_handles)
    else:
        splits = self.dataset_split_fn(self.dataset_or_dict, training_worker_handles)
        if not len(splits) == len(training_worker_handles):
            raise RuntimeError(f'The list of Datasets returned by the `dataset_split_fn`: {len(splits)} does not match the number of training workers: {len(training_worker_handles)}')
        return splits