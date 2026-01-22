import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def count_under_and_over_fits(self, overfit_border=0.15, underfit_border=0.95):
    """

        :param overfit_border: min fraction of iterations until overfitting starts one expects all models to have
        :param underfit_border: border, after which there should be no best_metric_scores
        :return: #models with best_metric > underfit_border * iter_count, #models, with best_metric > overfit_border
        """
    count_overfitting = 0
    count_underfitting = 0
    for fold_id, fold_curve in self._fold_curves.items():
        best_score_position = self._fold_metric_iteration[fold_id]
        best_model_size_fraction = best_score_position * 1.0 / len(fold_curve)
        if best_model_size_fraction > overfit_border:
            count_underfitting += 1
        elif best_model_size_fraction < underfit_border:
            count_overfitting += 1
    return (count_overfitting, count_underfitting)