import base64
import pyarrow.parquet.encryption as pe
def verify_file_encrypted(path):
    """Verify that the file is encrypted by looking at its first 4 bytes.
    If it's the magic string PARE
    then this is a parquet with encrypted footer."""
    with open(path, 'rb') as file:
        magic_str = file.read(4)
        assert magic_str == b'PARE'