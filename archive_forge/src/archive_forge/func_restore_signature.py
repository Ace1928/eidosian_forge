import contextlib
import functools
import pluggy
import keyring.errors
def restore_signature(func):

    @functools.wraps(func)
    def wrapper(url, username):
        return func(url, username)
    return wrapper