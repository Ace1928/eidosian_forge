from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
def test_annotationbbox_properties():
    ab = AnnotationBbox(DrawingArea(20, 20, 0, 0, clip=True), (0.5, 0.5), xycoords='data')
    assert ab.xyann == (0.5, 0.5)
    assert ab.anncoords == 'data'
    ab = AnnotationBbox(DrawingArea(20, 20, 0, 0, clip=True), (0.5, 0.5), xybox=(-0.2, 0.4), xycoords='data', boxcoords='axes fraction')
    assert ab.xyann == (-0.2, 0.4)
    assert ab.anncoords == 'axes fraction'