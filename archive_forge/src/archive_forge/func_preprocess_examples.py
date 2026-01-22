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
def preprocess_examples(self, texts: Union[TextInput, List[TextInput]], images: ImageInput=None, bboxes: BboxInput=None, num_image_tokens: Optional[int]=64) -> Union[str, List[str]]:
    """Add image and bounding box information to `texts` as image and patch index tokens.

        Args:
            texts (`Union[TextInput, List[TextInput]]`): The texts to be processed.
            images (`ImageInput`, *optional*): The images associated to `texts`.
            bboxes (`Union[List[Tuple[int]], List[Tuple[float]], List[List[Tuple[int]]], List[List[Tuple[float]]]]`, *optional*):
                The bounding bboxes associated to `texts`.
            num_image_tokens (`int`, *optional*, defaults to 64):
                The number of image tokens (used as latent queries). This should corresponds to the `latent_query_num`
                attribute in `Kosmos2Config`.

        Returns:
            `Union[TextInput, List[TextInput]]`: The processed texts with image and patch index tokens.
        """
    img_tokens = [self.boi_token] * num_image_tokens
    img_info_tokens = ' '.join([self.boi_token] + img_tokens + [self.eoi_token])
    batched = True
    if isinstance(texts, str):
        batched = False
        texts = [texts]
    if images is None:
        images = [None] * len(texts)
    elif not is_batched(images):
        images = [images]
    if len(texts) != len(images):
        raise ValueError(f'The number of examples in `texts` and `images` should be the same. Got {len(texts)} v.s. {len(images)} instead.')
    if not batched:
        self._check_bboxes_for_single_text(bboxes)
        bboxes = [bboxes]
    elif bboxes is not None:
        if not isinstance(bboxes, list):
            raise ValueError('`bboxes` should be `None` or a list (as a batch) when `texts` is passed as a batch.')
        for x in bboxes:
            self._check_bboxes_for_single_text(x)
    else:
        bboxes = [None] * len(texts)
    if len(bboxes) != len(texts):
        raise ValueError(f'The number of examples in `texts` and `bboxes` should be the same. Got {len(texts)} v.s. {len(bboxes)} instead.')
    result = [self._preprocess_single_example(text, image, bbox, img_info_tokens) for text, image, bbox in zip(texts, images, bboxes)]
    if not batched:
        result = result[0]
    return result