import _thread
import json
import logging
import random
import time
import typing
from redis import client
from . import exceptions, utils
@property
def client_name(self):
    return f'{self.channel}-lock'