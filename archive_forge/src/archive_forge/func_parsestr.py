from io import StringIO, TextIOWrapper
from email.feedparser import FeedParser, BytesFeedParser
from email._policybase import compat32
def parsestr(self, text, headersonly=True):
    return Parser.parsestr(self, text, True)