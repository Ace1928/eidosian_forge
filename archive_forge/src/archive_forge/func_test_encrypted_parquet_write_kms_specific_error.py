import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_write_kms_specific_error(tempdir, data_table, basic_encryption_config):
    """Write an encrypted parquet, but raise KeyError in KmsClient."""
    path = tempdir / 'encrypted_table_kms_error.in_mem.parquet'
    encryption_config = basic_encryption_config
    kms_connection_config = pe.KmsConnectionConfig()

    class ThrowingKmsClient(pe.KmsClient):
        """A KmsClient implementation that throws exception in
        wrap/unwrap calls
        """

        def __init__(self, config):
            """Create an InMemoryKmsClient instance."""
            pe.KmsClient.__init__(self)
            self.config = config

        def wrap_key(self, key_bytes, master_key_identifier):
            raise ValueError('Cannot Wrap Key')

        def unwrap_key(self, wrapped_key, master_key_identifier):
            raise ValueError('Cannot Unwrap Key')

    def kms_factory(kms_connection_configuration):
        return ThrowingKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    with pytest.raises(ValueError, match='Cannot Wrap Key'):
        write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)