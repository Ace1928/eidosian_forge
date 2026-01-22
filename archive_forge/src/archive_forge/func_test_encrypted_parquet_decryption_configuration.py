import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_decryption_configuration():
    decryption_config = pe.DecryptionConfiguration(cache_lifetime=timedelta(minutes=10.0))
    assert timedelta(minutes=10.0) == decryption_config.cache_lifetime
    decryption_config_1 = pe.DecryptionConfiguration()
    decryption_config_1.cache_lifetime = timedelta(minutes=10.0)
    assert timedelta(minutes=10.0) == decryption_config_1.cache_lifetime