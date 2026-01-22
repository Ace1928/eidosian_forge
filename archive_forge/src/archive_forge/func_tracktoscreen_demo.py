import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def tracktoscreen_demo(track='taylor swift', limit=10):
    """
    Track keywords from the public Streaming API and send output to terminal.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.filter(track=track)