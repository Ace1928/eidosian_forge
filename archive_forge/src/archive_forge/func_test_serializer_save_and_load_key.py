import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_serializer_save_and_load_key(self, tmp_path):
    input_key = {'test_key': 'key_val'}
    Serializer.save(self.image1, tmp_path / 'test_serializer_save_and_load_data.pkl', key=input_key)
    key = Deserializer.key(tmp_path / 'test_serializer_save_and_load_data.pkl')
    assert key == input_key