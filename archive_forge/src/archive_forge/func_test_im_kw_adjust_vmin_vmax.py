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
def test_im_kw_adjust_vmin_vmax(pyplot):
    """Check that im_kw passes kwargs to imshow"""
    confusion_matrix = np.array([[0.48, 0.04], [0.08, 0.4]])
    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot(im_kw=dict(vmin=0.0, vmax=0.8))
    clim = disp.im_.get_clim()
    assert clim[0] == pytest.approx(0.0)
    assert clim[1] == pytest.approx(0.8)