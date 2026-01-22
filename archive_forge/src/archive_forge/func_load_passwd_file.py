import os
import re
import logging
import collections
import pyzor.account
def load_passwd_file(passwd_fn):
    """Load the accounts from the specified file.

    Each line of the file should be in the format:
        username : key

    If the file does not exist, then an empty dictionary is returned;
    otherwise, a dictionary of (username, key) items is returned.
    """
    log = logging.getLogger('pyzord')
    accounts = {}
    if not os.path.exists(passwd_fn):
        log.info('Accounts file does not exist - only the anonymous user will be available.')
        return accounts
    passwdf = open(passwd_fn)
    for line in passwdf:
        if not line.strip() or line[0] == '#':
            continue
        try:
            user, key = line.split(':')
        except ValueError:
            log.warn('Invalid accounts line: %r', line)
            continue
        user = user.strip()
        key = key.strip()
        log.debug('Creating an account for %s with key %s.', user, key)
        accounts[user] = key
    passwdf.close()
    log.info('Accounts: %s', ','.join(accounts))
    return accounts