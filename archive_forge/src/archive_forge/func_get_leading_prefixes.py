import re
import time
import platform
from collections import OrderedDict
import six
def get_leading_prefixes(sequences):
    """
    Return a set of proper prefixes for given sequence of strings.

    :arg iterable sequences
    :rtype: set
    :return: Set of all string prefixes

    Given an iterable of strings, all textparts leading up to the final
    string is returned as a unique set.  This function supports the
    :meth:`~.Terminal.inkey` method by determining whether the given
    input is a sequence that **may** lead to a final matching pattern.

    >>> prefixes(['abc', 'abdf', 'e', 'jkl'])
    set([u'a', u'ab', u'abd', u'j', u'jk'])
    """
    return {seq[:i] for seq in sequences for i in range(1, len(seq))}