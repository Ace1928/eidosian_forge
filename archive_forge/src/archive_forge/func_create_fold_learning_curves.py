import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def create_fold_learning_curves(self, fold, offset=None):
    """

        :param fold: FoldId to plot
        :param offset: first iteration to plot
        :return: plotly figure for all cases on specified fold
        """
    import plotly.graph_objs as go
    traces = []
    for case in self.get_cases():
        case_result = self.get_case_result(case)
        scores_curve = case_result.get_fold_curve(fold)
        if offset is not None:
            first_idx = offset
        else:
            first_idx = int(len(scores_curve) * 0.1)
        traces.append(go.Scatter(x=[i * int(case_result.get_eval_step()) for i in range(first_idx, len(scores_curve))], y=scores_curve[first_idx:], mode='lines', name='Case {}'.format(case)))
    layout = go.Layout(title='Learning curves for metric {} on fold #{}'.format(self._metric_description, fold), hovermode='closest', xaxis=dict(title='Iteration', ticklen=5, zeroline=False, gridwidth=2), yaxis=dict(title='Metric', ticklen=5, gridwidth=2), showlegend=True)
    fig = go.Figure(data=traces, layout=layout)
    return fig