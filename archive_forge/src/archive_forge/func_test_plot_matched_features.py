import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
@pytest.mark.skipif(not has_mpl, reason='Matplotlib not installed')
@pytest.mark.parametrize('shapes', [((10, 10), (10, 10)), ((10, 10), (12, 10)), ((10, 10), (10, 12)), ((10, 10), (12, 12)), ((12, 10), (10, 10)), ((10, 12), (10, 10)), ((12, 12), (10, 10))])
def test_plot_matched_features(shapes):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    keypoints0 = 10 * np.random.rand(10, 2)
    keypoints1 = 10 * np.random.rand(10, 2)
    idxs0 = np.random.randint(10, size=10)
    idxs1 = np.random.randint(10, size=10)
    matches = np.column_stack((idxs0, idxs1))
    shape0, shape1 = shapes
    img0 = np.zeros(shape0)
    img1 = np.zeros(shape1)
    plot_matched_features(img0, img1, keypoints0=keypoints0, keypoints1=keypoints1, matches=matches, ax=ax)
    plot_matched_features(img0, img1, ax=ax, keypoints0=keypoints0, keypoints1=keypoints1, matches=matches, only_matches=True)
    plot_matched_features(img0, img1, ax=ax, keypoints0=keypoints0, keypoints1=keypoints1, matches=matches, keypoints_color='r')
    plot_matched_features(img0, img1, ax=ax, keypoints0=keypoints0, keypoints1=keypoints1, matches=matches, matches_color='r')
    plot_matched_features(img0, img1, ax=ax, keypoints0=keypoints0, keypoints1=keypoints1, matches=matches, alignment='vertical')
    plt.close()