from __future__ import absolute_import, division, print_function
from unicodedata import normalize
from ansible.errors import AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.six import text_type
def unicode_normalize(data, form='NFC'):
    """Applies normalization to 'unicode' strings.

    Args:
        data: A unicode string piped into the Jinja filter
        form: One of ('NFC', 'NFD', 'NFKC', 'NFKD').
              See https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize for more information.

    Returns:
        A normalized unicode string of the specified 'form'.
    """
    if not isinstance(data, text_type):
        raise AnsibleFilterTypeError('%s is not a valid input type' % type(data))
    if form not in ('NFC', 'NFD', 'NFKC', 'NFKD'):
        raise AnsibleFilterError('%s is not a valid form' % form)
    return normalize(form, data)