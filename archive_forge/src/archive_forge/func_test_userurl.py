from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_userurl(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.userurl.csv.ref'
    outfn = tmp_path / 'tweets.20150430-223406.userurl.csv'
    json2csv_entities(infile, outfn, ['id', 'screen_name'], 'user.urls', ['url', 'expanded_url'], gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)