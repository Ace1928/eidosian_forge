import pytest
from ase.utils.filecache import MultiFileJSONCache, CombinedJSONCache, Locked
def test_no_overwrite_combine(cache):
    cache.combine()
    with pytest.raises(RuntimeError, match='Already exists'):
        cache.combine()