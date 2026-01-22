import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_layout(self, tmp_path):
    Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_layout', info={'info': 'example'}, key={1: 2})