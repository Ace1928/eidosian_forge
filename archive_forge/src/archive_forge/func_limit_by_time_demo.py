import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def limit_by_time_demo(keywords='nltk'):
    """
    Query the REST API for Tweets about NLTK since yesterday and send
    the output to terminal.

    This example makes the assumption that there are sufficient Tweets since
    yesterday for the date to be an effective cut-off.
    """
    date = yesterday()
    dt_date = datetime.datetime(*date)
    oauth = credsfromfile()
    client = Query(**oauth)
    client.register(TweetViewer(limit=100, lower_date_limit=date))
    print(f'Cutoff date: {dt_date}\n')
    for tweet in client.search_tweets(keywords=keywords):
        print('{} '.format(tweet['created_at']), end='')
        client.handler.handle(tweet)