import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def expand_tweetids_demo():
    """
    Given a file object containing a list of Tweet IDs, fetch the
    corresponding full Tweets, if available.

    """
    ids_f = StringIO('        588665495492124672\n        588665495487909888\n        588665495508766721\n        588665495513006080\n        588665495517200384\n        588665495487811584\n        588665495525588992\n        588665495487844352\n        588665495492014081\n        588665495512948737')
    oauth = credsfromfile()
    client = Query(**oauth)
    hydrated = client.expand_tweetids(ids_f)
    for tweet in hydrated:
        id_str = tweet['id_str']
        print(f'id: {id_str}')
        text = tweet['text']
        if text.startswith('@null'):
            text = '[Tweet not available]'
        print(text + '\n')