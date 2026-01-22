import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_load_layout_entry1(self, tmp_path):
    Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entry1', info={'info': 'example'}, key={1: 2})
    entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry1.hvz')
    assert 'Image.I' in entries, "Entry 'Image.I' missing"
    loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry1.hvz', entries=['Image.I'])
    assert_element_equal(loaded, self.image1)