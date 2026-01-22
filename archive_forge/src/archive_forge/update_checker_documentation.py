import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
Return a UpdateResult object if there is a newer version.