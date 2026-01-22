from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_retweet_original_tweet(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.retweet.csv.ref'
    outfn = tmp_path / 'tweets.20150430-223406.retweet.csv'
    json2csv_entities(infile, outfn, ['id'], 'retweeted_status', ['created_at', 'favorite_count', 'id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweet_count', 'text', 'truncated', 'user.id'], gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)