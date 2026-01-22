import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def followtoscreen_demo(limit=10):
    """
    Using the Streaming API, select just the tweets from a specified list of
    userIDs.

    This is will only give results in a reasonable time if the users in
    question produce a high volume of tweets, and may even so show some delay.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.statuses.filter(follow=USERIDS)