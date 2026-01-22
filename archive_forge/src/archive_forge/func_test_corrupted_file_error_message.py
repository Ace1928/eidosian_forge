from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
def test_corrupted_file_error_message(fetch_kddcup99_fxt, tmp_path):
    """Check that a nice error message is raised when cache is corrupted."""
    kddcup99_dir = tmp_path / 'kddcup99_10-py3'
    kddcup99_dir.mkdir()
    samples_path = kddcup99_dir / 'samples'
    with samples_path.open('wb') as f:
        f.write(b'THIS IS CORRUPTED')
    msg = f'The cache for fetch_kddcup99 is invalid, please delete {str(kddcup99_dir)} and run the fetch_kddcup99 again'
    with pytest.raises(OSError, match=msg):
        fetch_kddcup99_fxt(data_home=str(tmp_path))