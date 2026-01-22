import re
import datetime
import numpy as np
import csv
import ctypes
def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
    """
    Workaround for the bug https://bugs.python.org/issue30157 if is unpatched.
    """
    if csv_sniffer_has_bug_last_field():
        right_regex = '(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)'
        for restr in ('(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?P=delim)', '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)', right_regex, '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)'):
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            matches = regexp.findall(sniff_line)
            if matches:
                break
        if restr != right_regex:
            return
        groupindex = regexp.groupindex
        assert len(matches) == 1
        m = matches[0]
        n = groupindex['quote'] - 1
        quote = m[n]
        n = groupindex['delim'] - 1
        delim = m[n]
        n = groupindex['space'] - 1
        space = bool(m[n])
        dq_regexp = re.compile(f'(({re.escape(delim)})|^)\\W*{quote}[^{re.escape(delim)}\\n]*{quote}[^{re.escape(delim)}\\n]*{quote}\\W*(({re.escape(delim)})|$)', re.MULTILINE)
        doublequote = bool(dq_regexp.search(sniff_line))
        dialect.quotechar = quote
        if delim in delimiters:
            dialect.delimiter = delim
        dialect.doublequote = doublequote
        dialect.skipinitialspace = space