import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def short_periodical_tasks():
    for i in range(10):
        client.add(increment=1)
        time.sleep(1)