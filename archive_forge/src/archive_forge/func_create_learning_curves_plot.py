import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def create_learning_curves_plot(self, offset=None):
    """

        :param offset: First iteration to plot
        :return: plotly Figure with learning curves for each fold
        """
    import plotly.graph_objs as go
    traces = []
    for fold in self.get_fold_ids():
        scores_curve = self.get_fold_curve(fold)
        if offset is not None:
            first_idx = offset
        else:
            first_idx = int(len(scores_curve) * 0.1)
        traces.append(go.Scatter(x=[i * int(self._eval_step) for i in range(first_idx, len(scores_curve))], y=scores_curve[first_idx:], mode='lines', name='Fold #{}'.format(fold)))
    layout = go.Layout(title='Learning curves for case {}'.format(self._case), hovermode='closest', xaxis=dict(title='Iteration', ticklen=5, zeroline=False, gridwidth=2), yaxis=dict(title='Metric', ticklen=5, gridwidth=2), showlegend=True)
    fig = go.Figure(data=traces, layout=layout)
    return fig