import os
import tempfile
import testtools
from troveclient import utils
def test_encode_decode_data(self):
    text_data_str = 'This is a text string'
    try:
        text_data_bytes = bytes('This is a byte stream', 'utf-8')
    except TypeError:
        text_data_bytes = bytes('This is a byte stream')
    random_data_str = os.urandom(12)
    random_data_bytes = bytearray(os.urandom(12))
    special_char_str = '\x00ÿ\x00ÿÿ\x00'
    special_char_bytes = bytearray([ord(item) for item in special_char_str])
    data = [text_data_str, text_data_bytes, random_data_str, random_data_bytes, special_char_str, special_char_bytes]
    for datum in data:
        try:
            expected_deserialized = bytearray([ord(item) for item in datum])
        except TypeError:
            expected_deserialized = bytearray([item for item in datum])
        serialized_data = utils.encode_data(datum)
        self.assertIsNotNone(serialized_data, "'%s' serialized is None" % datum)
        deserialized_data = utils.decode_data(serialized_data)
        self.assertIsNotNone(deserialized_data, "'%s' deserialized is None" % datum)
        self.assertEqual(expected_deserialized, deserialized_data, 'Serialize/Deserialize failed')
        with tempfile.NamedTemporaryFile() as temp_file:
            with open(temp_file.name, 'wb') as fh_w:
                fh_w.write(bytearray([ord(item) for item in serialized_data]))
            with open(temp_file.name, 'rb') as fh_r:
                new_serialized_data = fh_r.read()
            new_deserialized_data = utils.decode_data(new_serialized_data)
            self.assertIsNotNone(new_deserialized_data, "'%s' deserialized is None" % datum)
            self.assertEqual(expected_deserialized, new_deserialized_data, 'Serialize/Deserialize with files failed')