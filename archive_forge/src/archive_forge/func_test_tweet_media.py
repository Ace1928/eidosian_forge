from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_tweet_media(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.media.csv.ref'
    outfn = tmp_path / 'tweets.20150430-223406.media.csv'
    json2csv_entities(infile, outfn, ['id'], 'media', ['media_url', 'url'], gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)