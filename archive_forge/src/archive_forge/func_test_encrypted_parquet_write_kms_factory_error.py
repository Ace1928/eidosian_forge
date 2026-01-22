import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_write_kms_factory_error(tempdir, data_table, basic_encryption_config):
    """Write an encrypted parquet, but raise ValueError in kms_factory."""
    path = tempdir / 'encrypted_table_kms_factory_error.in_mem.parquet'
    encryption_config = basic_encryption_config
    kms_connection_config = pe.KmsConnectionConfig()

    def kms_factory(kms_connection_configuration):
        raise ValueError('Cannot create KmsClient')
    crypto_factory = pe.CryptoFactory(kms_factory)
    with pytest.raises(ValueError, match='Cannot create KmsClient'):
        write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)