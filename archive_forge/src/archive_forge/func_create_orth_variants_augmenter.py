import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
@registry.augmenters('spacy.orth_variants.v1')
def create_orth_variants_augmenter(level: float, lower: float, orth_variants: Dict[str, List[Dict]]) -> Callable[['Language', Example], Iterator[Example]]:
    """Create a data augmentation callback that uses orth-variant replacement.
    The callback can be added to a corpus or other data iterator during training.

    level (float): The percentage of texts that will be augmented.
    lower (float): The percentage of texts that will be lowercased.
    orth_variants (Dict[str, List[Dict]]): A dictionary containing
        the single and paired orth variants. Typically loaded from a JSON file.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(orth_variants_augmenter, orth_variants=orth_variants, level=level, lower=lower)