import sys
from unittest import mock
from stevedore import _cache
from stevedore.tests import utils
def test_disable_caching_file(self):
    """Test caching is disabled if .disable file is present in target
        dir
        """
    cache_dir = _cache._get_cache_dir()
    with mock.patch('os.path.isfile') as mock_path:
        mock_path.return_value = True
        sot = _cache.Cache()
        mock_path.assert_called_with('%s/.disable' % cache_dir)
        self.assertTrue(sot._disable_caching)
        mock_path.return_value = False
        sot = _cache.Cache()
        self.assertFalse(sot._disable_caching)