import datetime
import gzip
import itertools
import json
import os
import time
import requests
from twython import Twython, TwythonStreamer
from twython.exceptions import TwythonError, TwythonRateLimitError
from nltk.twitter.api import BasicTweetHandler, TweetHandlerI
from nltk.twitter.util import credsfromfile, guess_path
def user_tweets(self, screen_name, limit, include_rts='false'):
    """
        Return a collection of the most recent Tweets posted by the user

        :param str user: The user's screen name; the initial '@' symbol        should be omitted
        :param int limit: The number of Tweets to recover; 200 is the maximum allowed
        :param str include_rts: Whether to include statuses which have been        retweeted by the user; possible values are 'true' and 'false'
        """
    data = self.get_user_timeline(screen_name=screen_name, count=limit, include_rts=include_rts)
    for item in data:
        self.handler.handle(item)