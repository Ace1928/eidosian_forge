import sys, re, curl, exceptions
from the command line first, then standard input.
def get_MAC_address(self, page, prefix):
    return self.screen_scrape('', prefix + ':[^M]*\\(MAC Address: *([^)]*)')