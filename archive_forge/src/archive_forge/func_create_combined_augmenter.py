import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
@registry.augmenters('spacy.combined_augmenter.v1')
def create_combined_augmenter(lower_level: float, orth_level: float, orth_variants: Optional[Dict[str, List[Dict]]], whitespace_level: float, whitespace_per_token: float, whitespace_variants: Optional[List[str]]) -> Callable[['Language', Example], Iterator[Example]]:
    """Create a data augmentation callback that uses orth-variant replacement.
    The callback can be added to a corpus or other data iterator during training.

    lower_level (float): The percentage of texts that will be lowercased.
    orth_level (float): The percentage of texts that will be augmented.
    orth_variants (Optional[Dict[str, List[Dict]]]): A dictionary containing the
        single and paired orth variants. Typically loaded from a JSON file.
    whitespace_level (float): The percentage of texts that will have whitespace
        tokens inserted.
    whitespace_per_token (float): The number of whitespace tokens to insert in
        the modified doc as a percentage of the doc length.
    whitespace_variants (Optional[List[str]]): The whitespace token texts.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(combined_augmenter, lower_level=lower_level, orth_level=orth_level, orth_variants=orth_variants, whitespace_level=whitespace_level, whitespace_per_token=whitespace_per_token, whitespace_variants=whitespace_variants)