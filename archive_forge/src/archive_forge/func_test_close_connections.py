from __future__ import print_function
import boto
import time
import uuid
from threading import Thread
def test_close_connections():
    """
    A test that exposes the problem where connections are returned to the
    connection pool (and closed) before the caller reads the response.
    
    I couldn't think of a way to test it without greenlets, so this test
    doesn't run as part of the standard test suite.  That way, no more
    dependencies are added to the test suite.
    """
    print('Running test_close_connections')
    s3 = boto.connect_s3()
    for b in s3.get_all_buckets():
        if b.name.startswith('test-'):
            for key in b.get_all_keys():
                key.delete()
            b.delete()
    bucket = s3.create_bucket('test-%d' % int(time.time()))
    names = [str(uuid.uuid4) for _ in range(30)]
    threads = [spawn(put_object, bucket, name) for name in names]
    for t in threads:
        t.join()
    threads = [spawn(get_object, bucket, name) for name in names]
    for t in threads:
        t.join()