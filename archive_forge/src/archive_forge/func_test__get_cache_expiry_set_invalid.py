import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
@mock.patch.object(filecache, 'CACHE', None)
@mock.patch.object(filecache.LOG, 'warning', autospec=True)
@mock.patch.object(os.environ, 'get', autospec=True)
@mock.patch.object(os.path, 'exists', autospec=True)
@mock.patch.object(os, 'makedirs', autospec=True)
@mock.patch.object(dogpile.cache, 'make_region', autospec=True)
def test__get_cache_expiry_set_invalid(self, mock_makeregion, mock_makedirs, mock_exists, mock_get, mock_log):
    cache_val = 5643
    cache_expiry = 'Rollenhagen'
    mock_get.return_value = cache_expiry
    mock_exists.return_value = False
    cache_region = mock.Mock(spec=dogpile.cache.region.CacheRegion)
    cache_region.configure.return_value = cache_val
    mock_makeregion.return_value = cache_region
    self.assertEqual(cache_val, filecache._get_cache())
    self.assertEqual(cache_val, filecache.CACHE)
    mock_get.assert_called_once_with(filecache.CACHE_EXPIRY_ENV_VAR, mock.ANY)
    cache_region.configure.assert_called_once_with(mock.ANY, arguments=mock.ANY, expiration_time=filecache.DEFAULT_EXPIRY)
    log_dict = {'curr_val': cache_expiry, 'default': filecache.DEFAULT_EXPIRY, 'env_var': filecache.CACHE_EXPIRY_ENV_VAR}
    mock_log.assert_called_once_with(mock.ANY, log_dict)