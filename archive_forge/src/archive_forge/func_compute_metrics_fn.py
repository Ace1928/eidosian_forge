import os
from typing import Callable, Dict
import numpy as np
from transformers import EvalPrediction
from transformers import glue_compute_metrics, glue_output_modes
def compute_metrics_fn(p: EvalPrediction):
    if output_mode == 'classification':
        preds = np.argmax(p.predictions, axis=1)
    elif output_mode == 'regression':
        preds = np.squeeze(p.predictions)
    metrics = glue_compute_metrics(task_name, preds, p.label_ids)
    return metrics