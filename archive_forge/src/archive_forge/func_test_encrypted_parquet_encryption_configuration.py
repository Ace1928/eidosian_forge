import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_encryption_configuration():

    def validate_encryption_configuration(encryption_config):
        assert FOOTER_KEY_NAME == encryption_config.footer_key
        assert ['a', 'b'] == encryption_config.column_keys[COL_KEY_NAME]
        assert 'AES_GCM_CTR_V1' == encryption_config.encryption_algorithm
        assert encryption_config.plaintext_footer
        assert not encryption_config.double_wrapping
        assert timedelta(minutes=10.0) == encryption_config.cache_lifetime
        assert not encryption_config.internal_key_material
        assert 192 == encryption_config.data_key_length_bits
    encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, column_keys={COL_KEY_NAME: ['a', 'b']}, encryption_algorithm='AES_GCM_CTR_V1', plaintext_footer=True, double_wrapping=False, cache_lifetime=timedelta(minutes=10.0), internal_key_material=False, data_key_length_bits=192)
    validate_encryption_configuration(encryption_config)
    encryption_config_1 = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME)
    encryption_config_1.column_keys = {COL_KEY_NAME: ['a', 'b']}
    encryption_config_1.encryption_algorithm = 'AES_GCM_CTR_V1'
    encryption_config_1.plaintext_footer = True
    encryption_config_1.double_wrapping = False
    encryption_config_1.cache_lifetime = timedelta(minutes=10.0)
    encryption_config_1.internal_key_material = False
    encryption_config_1.data_key_length_bits = 192
    validate_encryption_configuration(encryption_config_1)