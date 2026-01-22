import logging
import os.path
def close_both(*args):
    nonlocal inner
    try:
        outer_close()
    finally:
        if inner:
            inner, fp = (None, inner)
            fp.close()