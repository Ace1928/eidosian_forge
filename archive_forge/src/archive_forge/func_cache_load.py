import sys, re, curl, exceptions
from the command line first, then standard input.
def cache_load(self, page):
    if page not in self.pagecache:
        fetch = curl.Curl(self.host)
        fetch.set_verbosity(self.verbosity)
        fetch.get(page)
        self.pagecache[page] = fetch.body()
        if fetch.answered('401'):
            raise LinksysError('authorization failure.', True)
        elif not fetch.answered(LinksysSession.check_strings[page]):
            del self.pagecache[page]
            raise LinksysError('check string for page %s missing!' % os.path.join(self.host, page), False)
        fetch.close()