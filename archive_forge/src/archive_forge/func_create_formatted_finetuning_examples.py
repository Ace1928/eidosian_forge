import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def create_formatted_finetuning_examples(training_examples: Sequence[TrainingExample], formatter_configs: FormatterConfigs) -> List[str]:
    """
    This formatter takes consumer-provided training examples and converts them to
    the right format for finetuning llama-guard.

    There are various configuration options available.

    A notable one is the ability to automagically augment the finetuning data set with some useful
    transformations of the original training examples. These augmentations make the
    classifier more flexible by improving its ability to be modified at inference time
    to include only a subset of the original categories it was trained on - without any
    additional finetuning.

    Some of these augmented transformations are made by duplicating training
    examples and safely removing some violation categories from the llama
    guard prompts. Because of this, in some of this file you will see
    references to "original" category indices/codes and rewritten ones. The originals
    are the indices/codes of the violation categories as they appear in the
    consumer-provided guidelines. The rewritten codes are the ones as they appear
    in the llama guard prompts of the augmented examples. We occasionally need to
    convert between the two.
    """
    _verify_formatter_configs(formatter_configs)
    random.seed(formatter_configs.random_seed)
    indices_of_all_categories = range(len(formatter_configs.guidelines.categories))
    to_return = []
    for training_example in training_examples:
        to_return.append(_create_formatted_finetuning_example(training_example, formatter_configs, category_indices_to_include_in_llama_guard_prompt=list(indices_of_all_categories)))
        _maybe_add_data_augmentations_for_example(training_example, to_return, indices_of_all_categories, formatter_configs)
    return to_return