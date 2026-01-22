import collections
import urllib.parse
import urllib.request
def request_rate(self, useragent):
    if not self.mtime():
        return None
    for entry in self.entries:
        if entry.applies_to(useragent):
            return entry.req_rate
    if self.default_entry:
        return self.default_entry.req_rate
    return None