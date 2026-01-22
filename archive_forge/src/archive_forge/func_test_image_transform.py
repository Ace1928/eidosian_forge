import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_transform(self):
    img = Image(np.random.rand(10, 10))
    op_img = transform(img, operator=lambda x: x * 2)
    self.assertEqual(op_img, img.clone(img.data * 2, group='Transform'))