import copy
import math
import re
from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
def extract_entities_with_patch_indices(text):
    """Extract entities contained in `text`. The bounding bboxes is given in the form of patch indices.

    This functioin is only intended to be used within `clean_text_and_extract_entities_with_bboxes` where further
    processing happens, including converting to normalized coordinates and whitespace character cleaning up.

    Examples:

    ```python
    >>> text = "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
    >>> entities = extract_entities_with_patch_indices(text)
    >>> entities
    [(' a snowman', (31, 41), [(44, 863)]), (' a fire', (130, 137), [(5, 911)])]
    ```"""
    pattern = '(?:(<phrase>([^<]+)</phrase>))?<object>((?:<patch_index_\\d+><patch_index_\\d+></delimiter_of_multi_objects/>)*<patch_index_\\d+><patch_index_\\d+>)</object>'
    matches = re.finditer(pattern, text)
    entities_with_patch_indices = []
    for match in matches:
        span = match.span(2)
        phrase_tag, phrase, match_content = match.groups()
        if not phrase_tag:
            phrase = None
            span = (match.span(0)[0], match.span(0)[0])
        patch_index_pairs = match_content.split('</delimiter_of_multi_objects/>')
        entity_bboxes = []
        for pair in patch_index_pairs:
            x = re.search('<patch_index_(\\d+)>', pair)
            y = re.search('<patch_index_(\\d+)>', pair[1:])
            if x and y:
                if phrase:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1))))
                else:
                    entity_bboxes.append((int(x.group(1)), int(y.group(1))))
        if phrase:
            entities_with_patch_indices.append((phrase, span, entity_bboxes))
        else:
            for bbox in entity_bboxes:
                entity = f'<patch_index_{bbox[0]}><patch_index_{bbox[1]}>'
                entities_with_patch_indices.append((entity, span, [bbox]))
    return entities_with_patch_indices