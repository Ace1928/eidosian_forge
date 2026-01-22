import os
import warnings
import re
def getcaps():
    """Return a dictionary containing the mailcap database.

    The dictionary maps a MIME type (in all lowercase, e.g. 'text/plain')
    to a list of dictionaries corresponding to mailcap entries.  The list
    collects all the entries for that MIME type from all available mailcap
    files.  Each dictionary contains key-value pairs for that MIME type,
    where the viewing command is stored with the key "view".

    """
    caps = {}
    lineno = 0
    for mailcap in listmailcapfiles():
        try:
            fp = open(mailcap, 'r')
        except OSError:
            continue
        with fp:
            morecaps, lineno = _readmailcapfile(fp, lineno)
        for key, value in morecaps.items():
            if not key in caps:
                caps[key] = value
            else:
                caps[key] = caps[key] + value
    return caps