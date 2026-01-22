import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
the server thread exits even if there are no connections