import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
@property
def serialization_data(self):
    raise NotImplementedError('Cannot serialize Developer Shell credentials.')