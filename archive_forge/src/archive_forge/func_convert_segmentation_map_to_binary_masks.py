import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def convert_segmentation_map_to_binary_masks(self, segmentation_map: 'np.ndarray', instance_id_to_semantic_id: Optional[Dict[int, int]]=None, ignore_index: Optional[int]=None, reduce_labels: bool=False):
    reduce_labels = reduce_labels if reduce_labels is not None else self.reduce_labels
    ignore_index = ignore_index if ignore_index is not None else self.ignore_index
    return convert_segmentation_map_to_binary_masks(segmentation_map=segmentation_map, instance_id_to_semantic_id=instance_id_to_semantic_id, ignore_index=ignore_index, reduce_labels=reduce_labels)