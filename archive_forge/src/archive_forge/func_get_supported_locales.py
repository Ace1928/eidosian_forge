import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES
from typing import Iterable, Any, Union, Dict, Optional
def get_supported_locales() -> Iterable[str]:
    """Returns a list of all the supported locale codes."""
    return _supported_locales