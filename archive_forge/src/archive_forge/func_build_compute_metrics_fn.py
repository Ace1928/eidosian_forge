import os
from typing import Callable, Dict
import numpy as np
from transformers import EvalPrediction
from transformers import glue_compute_metrics, glue_output_modes
def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    """Function from transformers/examples/text-classification/run_glue.py"""
    output_mode = glue_output_modes[task_name]

    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == 'classification':
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == 'regression':
            preds = np.squeeze(p.predictions)
        metrics = glue_compute_metrics(task_name, preds, p.label_ids)
        return metrics
    return compute_metrics_fn