import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_read_metadata_no_decryption_config(tempdir, data_table):
    """Write an encrypted parquet, verify it's encrypted,
    but then try to read its metadata without decryption properties."""
    test_encrypted_parquet_write_read(tempdir, data_table)
    with pytest.raises(IOError, match='no decryption'):
        pq.read_metadata(tempdir / PARQUET_NAME)