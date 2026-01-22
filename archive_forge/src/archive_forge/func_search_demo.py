import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def search_demo(keywords='nltk'):
    """
    Use the REST API to search for past tweets containing a given keyword.
    """
    oauth = credsfromfile()
    client = Query(**oauth)
    for tweet in client.search_tweets(keywords=keywords, limit=10):
        print(tweet['text'])