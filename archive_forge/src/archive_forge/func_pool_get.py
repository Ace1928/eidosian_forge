from __future__ import print_function
import sys
import time
import urllib
import urllib3  # noqa: E402
def pool_get(url_list):
    assert url_list
    pool = urllib3.PoolManager()
    for url in url_list:
        now = time.time()
        pool.request('GET', url, assert_same_host=False)
        elapsed = time.time() - now
        print('Got in %0.3fs: %s' % (elapsed, url))