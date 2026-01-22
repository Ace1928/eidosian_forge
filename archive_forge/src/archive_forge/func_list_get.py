import logging
import re
def list_get(xs, i):
    try:
        return xs[i]
    except IndexError:
        return None