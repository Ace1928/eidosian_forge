from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def configure_worker(worker):
    """Configure individual TPU worker.

      Args:
        worker: A dict with the field ipAddress where the configure request will
          be sent.
      """
    ip_address = worker['ipAddress']
    url = (_VERSION_SWITCHER_ENDPOINT + '/{}?restartType={}').format(ip_address, version, restart_type)
    req = urllib.request.Request(url, data=b'')
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        status_code = e.code
        if status_code == 404:
            raise Exception('Tensorflow version {} is not available on Cloud TPU, try a previous nightly version or refer to https://cloud.google.com/tpu/docs/release-notes for the latest official version.'.format(version))
        else:
            raise Exception('Failed to configure worker {}'.format(ip_address))