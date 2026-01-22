import os
import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import pytest
from kivy.network.urlrequest import UrlRequestUrllib as UrlRequest
Passing a `ca_file` should not crash on http scheme, refs #6946