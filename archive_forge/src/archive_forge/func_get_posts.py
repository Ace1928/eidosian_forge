import bz2
import io
import json
import lzma
import os
import re
import requests
import subprocess
import zstandard as zstd
from bs4 import BeautifulSoup
from os.path import isfile
from os.path import join as pjoin
from time import sleep, time
from collections import defaultdict
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize
def get_posts(post_ids):
    posts = ','.join(post_ids)
    post_ids_link = f'https://api.pushshift.io/reddit/search/submission/?ids={posts}'
    req = requests.get(post_ids_link)
    post_json = json.loads(req.text)
    for post in post_json['data']:
        yield (post['subreddit'].lower(), json.dumps(post))