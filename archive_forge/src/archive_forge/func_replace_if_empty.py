import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def replace_if_empty(text, replacement='No relevant packages'):
    if text is not None and len(text) == 0:
        return replacement
    return text