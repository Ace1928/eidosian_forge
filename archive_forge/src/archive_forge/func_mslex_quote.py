import re
import shlex
from typing import List
from mlflow.utils.os import is_windows
def mslex_quote(s, for_cmd=True):
    """
    Quote a string for use as a command line argument in DOS or Windows.

    On windows, before a command line argument becomes a char* in a
    program's argv, it must be parsed by both cmd.exe, and by
    CommandLineToArgvW.

    If for_cmd is true, then this will quote the string so it will
    be parsed correctly by cmd.exe and then by CommandLineToArgvW.

    If for_cmd is false, then this will quote the string so it will
    be parsed correctly when passed directly to CommandLineToArgvW.

    For some strings there is no way to quote them so they will
    parse correctly in both situations.
    """
    if not s:
        return '""'
    if not re.search(cmd_meta_or_space, s):
        return s
    if for_cmd and re.search(cmd_meta, s):
        if not re.search(cmd_meta_inside_quotes, s):
            m = re.search('\\\\+$', s)
            if m:
                return '"' + s + m.group() + '"'
            else:
                return '"' + s + '"'
        if not re.search('[\\s\\"]', s):
            return re.sub(cmd_meta, '^\\1', s)
        return re.sub(cmd_meta, '^\\1', mslex_quote(s, for_cmd=False))
    i = re.finditer('(\\\\*)(\\"+)|(\\\\+)|([^\\\\\\"]+)', s)

    def parts():
        yield '"'
        for m in i:
            _, end = m.span()
            slashes, quotes, onlyslashes, text = m.groups()
            if quotes:
                yield slashes
                yield slashes
                yield ('\\"' * len(quotes))
            elif onlyslashes:
                if end == len(s):
                    yield onlyslashes
                    yield onlyslashes
                else:
                    yield onlyslashes
            else:
                yield text
        yield '"'
    return ''.join(parts())