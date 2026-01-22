import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def tm_to_coco(self, name: str='tm_map_input') -> None:
    """Utility function for converting the input for this metric to coco format and saving it to a json file.

        This function should be used after calling `.update(...)` or `.forward(...)` on all data that should be written
        to the file, as the input is then internally cached. The function then converts to information to coco format
        a writes it to json files.

        Args:
            name: Name of the output file, which will be appended with "_preds.json" and "_target.json"

        Example:
            >>> from torch import tensor
            >>> from torchmetrics.detection import MeanAveragePrecision
            >>> preds = [
            ...   dict(
            ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
            ...     scores=tensor([0.536]),
            ...     labels=tensor([0]),
            ...   )
            ... ]
            >>> target = [
            ...   dict(
            ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=tensor([0]),
            ...   )
            ... ]
            >>> metric = MeanAveragePrecision()
            >>> metric.update(preds, target)
            >>> metric.tm_to_coco("tm_map_input")  # doctest: +SKIP

        """
    target_dataset = self._get_coco_format(labels=self.groundtruth_labels, boxes=self.groundtruth_box, masks=self.groundtruth_mask, crowds=self.groundtruth_crowds, area=self.groundtruth_area)
    preds_dataset = self._get_coco_format(labels=self.detection_labels, boxes=self.detection_box, masks=self.detection_mask)
    preds_json = json.dumps(preds_dataset['annotations'], indent=4)
    target_json = json.dumps(target_dataset, indent=4)
    with open(f'{name}_preds.json', 'w') as f:
        f.write(preds_json)
    with open(f'{name}_target.json', 'w') as f:
        f.write(target_json)