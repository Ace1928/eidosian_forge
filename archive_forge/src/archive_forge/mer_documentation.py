from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
Match error rate is a metric of the performance of an automatic speech recognition system.

    This value indicates the percentage of words that were incorrectly predicted and inserted. The lower the value, the
    better the performance of the ASR system with a MatchErrorRate of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Match error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> match_error_rate(preds=preds, target=target)
        tensor(0.4444)

    