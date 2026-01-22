import re
from threading import Lock
from io import TextIOBase
from sqlparse import tokens, keywords
from sqlparse.utils import consume
def set_SQL_REGEX(self, SQL_REGEX):
    """Set the list of regex that will parse the SQL."""
    FLAGS = re.IGNORECASE | re.UNICODE
    self._SQL_REGEX = [(re.compile(rx, FLAGS).match, tt) for rx, tt in SQL_REGEX]