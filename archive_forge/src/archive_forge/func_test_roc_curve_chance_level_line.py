import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('plot_chance_level', [True, False])
@pytest.mark.parametrize('chance_level_kw', [None, {'linewidth': 1, 'color': 'red', 'label': 'DummyEstimator'}])
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_roc_curve_chance_level_line(pyplot, data_binary, plot_chance_level, chance_level_kw, constructor_name):
    """Check the chance level line plotting behaviour."""
    X, y = data_binary
    lr = LogisticRegression()
    lr.fit(X, y)
    y_pred = getattr(lr, 'predict_proba')(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    if constructor_name == 'from_estimator':
        display = RocCurveDisplay.from_estimator(lr, X, y, alpha=0.8, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw)
    else:
        display = RocCurveDisplay.from_predictions(y, y_pred, alpha=0.8, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw)
    import matplotlib as mpl
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_alpha() == 0.8
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert isinstance(display.figure_, mpl.figure.Figure)
    if plot_chance_level:
        assert isinstance(display.chance_level_, mpl.lines.Line2D)
        assert tuple(display.chance_level_.get_xdata()) == (0, 1)
        assert tuple(display.chance_level_.get_ydata()) == (0, 1)
    else:
        assert display.chance_level_ is None
    if plot_chance_level and chance_level_kw is None:
        assert display.chance_level_.get_color() == 'k'
        assert display.chance_level_.get_linestyle() == '--'
        assert display.chance_level_.get_label() == 'Chance level (AUC = 0.5)'
    elif plot_chance_level:
        assert display.chance_level_.get_label() == chance_level_kw['label']
        assert display.chance_level_.get_color() == chance_level_kw['color']
        assert display.chance_level_.get_linewidth() == chance_level_kw['linewidth']