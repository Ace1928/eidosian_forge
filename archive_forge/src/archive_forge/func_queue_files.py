import boto
from boto.services.service import Service
from boto.services.message import ServiceMessage
import os
import mimetypes
def queue_files(self):
    boto.log.info('Queueing files from %s' % self.input_bucket.name)
    for key in self.input_bucket:
        boto.log.info('Queueing %s' % key.name)
        m = ServiceMessage()
        if self.output_bucket:
            d = {'OutputBucket': self.output_bucket.name}
        else:
            d = None
        m.for_key(key, d)
        self.input_queue.write(m)