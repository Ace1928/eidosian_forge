import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_write_read_wrong_key(tempdir, data_table):
    """Write an encrypted parquet, verify it's encrypted,
    and then read it using wrong keys."""
    path = tempdir / PARQUET_NAME
    encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, column_keys={COL_KEY_NAME: ['a', 'b']}, encryption_algorithm='AES_GCM_V1', cache_lifetime=timedelta(minutes=5.0), data_key_length_bits=256)
    kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: FOOTER_KEY.decode('UTF-8'), COL_KEY_NAME: COL_KEY.decode('UTF-8')})

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)
    verify_file_encrypted(path)
    wrong_kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: COL_KEY.decode('UTF-8'), COL_KEY_NAME: FOOTER_KEY.decode('UTF-8')})
    decryption_config = pe.DecryptionConfiguration(cache_lifetime=timedelta(minutes=5.0))
    with pytest.raises(ValueError, match='Incorrect master key used'):
        read_encrypted_parquet(path, decryption_config, wrong_kms_connection_config, crypto_factory)