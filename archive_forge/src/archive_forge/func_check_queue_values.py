import os
import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import pytest
from kivy.network.urlrequest import UrlRequestUrllib as UrlRequest
def check_queue_values(queue):
    """Helper function verifying the queue contains the expected values."""
    assert len(queue) >= 3
    assert queue[0][1] == 'progress'
    assert queue[-2][1] == 'progress'
    assert queue[-1][1] in ('success', 'redirect')
    assert queue[0][2][0] == 0
    assert queue[-2][2][0] == queue[-2][2][1]