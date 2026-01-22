from typing import List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def word_information_lost(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Word Information Lost rate is a metric of the performance of an automatic speech recognition system.

    This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better
    the performance of the ASR system with a Word Information Lost rate of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Word Information Lost rate

    Examples:
        >>> from torchmetrics.functional.text import word_information_lost
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_information_lost(preds, target)
        tensor(0.6528)

    """
    errors, target_total, preds_total = _word_info_lost_update(preds, target)
    return _word_info_lost_compute(errors, target_total, preds_total)