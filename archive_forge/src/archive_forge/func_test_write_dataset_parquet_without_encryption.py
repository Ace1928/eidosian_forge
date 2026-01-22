from datetime import timedelta
import pyarrow.fs as fs
import pyarrow as pa
import pytest
@pytest.mark.skipif(not encryption_unavailable, reason='Parquet Encryption is currently enabled')
def test_write_dataset_parquet_without_encryption():
    """Test write_dataset with ParquetFileFormat and test if an exception is thrown
    if you try to set encryption_config using make_write_options"""
    pformat = pa.dataset.ParquetFileFormat()
    with pytest.raises(NotImplementedError):
        _ = pformat.make_write_options(encryption_config='some value')