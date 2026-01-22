import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def header_fetch_parse(self, name, value):
    """+
        If the value contains binary data, it is converted into a Header object
        using the unknown-8bit charset.  Otherwise it is returned unmodified.
        """
    return self._sanitize_header(name, value)