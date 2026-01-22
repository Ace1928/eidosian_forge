import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from ...models.bert.tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from ...utils import is_tf_available, is_torch_available, logging
from .utils import DataProcessor
def get_examples_from_dataset(self, dataset, evaluate=False):
    """
        Creates a list of [`~data.processors.squad.SquadExample`] using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from *tensorflow_datasets.load("squad")*
            evaluate: Boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples:

        ```python
        >>> import tensorflow_datasets as tfds

        >>> dataset = tfds.load("squad")

        >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
        >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        ```"""
    if evaluate:
        dataset = dataset['validation']
    else:
        dataset = dataset['train']
    examples = []
    for tensor_dict in tqdm(dataset):
        examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))
    return examples