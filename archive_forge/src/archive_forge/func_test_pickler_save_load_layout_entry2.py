import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_load_layout_entry2(self, tmp_path):
    Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entry2', info={'info': 'example'}, key={1: 2})
    entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry2.hvz')
    assert 'Image.II' in entries, "Entry 'Image.II' missing"
    loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry2.hvz', entries=['Image.II'])
    assert_element_equal(loaded, self.image2)