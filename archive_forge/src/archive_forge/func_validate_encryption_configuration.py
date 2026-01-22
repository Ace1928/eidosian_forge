import pytest
from datetime import timedelta
import pyarrow as pa
def validate_encryption_configuration(encryption_config):
    assert FOOTER_KEY_NAME == encryption_config.footer_key
    assert ['a', 'b'] == encryption_config.column_keys[COL_KEY_NAME]
    assert 'AES_GCM_CTR_V1' == encryption_config.encryption_algorithm
    assert encryption_config.plaintext_footer
    assert not encryption_config.double_wrapping
    assert timedelta(minutes=10.0) == encryption_config.cache_lifetime
    assert not encryption_config.internal_key_material
    assert 192 == encryption_config.data_key_length_bits