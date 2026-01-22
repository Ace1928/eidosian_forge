import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def setvars(self, **kw):
    """
        Creates a copy of this URL, but with all the variables set/reset
        (like .setvar(), except clears past variables at the same time)
        """
    return self.__class__(self.url, vars=kw.items(), attrs=self.attrs, params=self.original_params)