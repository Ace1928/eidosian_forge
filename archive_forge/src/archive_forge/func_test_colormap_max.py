import numpy as np
import pytest
from numpy.testing import (
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
def test_colormap_max(pyplot):
    """Check that the max color is used for the color of the text."""
    gray = pyplot.get_cmap('gray', 1024)
    confusion_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot(cmap=gray)
    color = disp.text_[1, 0].get_color()
    assert_allclose(color, [1.0, 1.0, 1.0, 1.0])