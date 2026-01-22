from __future__ import print_function
import boto
import time
import uuid
from threading import Thread
def read_big_object(s3, bucket, name, count):
    for _ in range(count):
        key = bucket.get_key(name)
        out = WriteAndCount()
        key.get_contents_to_file(out)
        if out.size != BIG_SIZE:
            print(out.size, BIG_SIZE)
        assert out.size == BIG_SIZE
        print('    pool size:', s3._pool.size())