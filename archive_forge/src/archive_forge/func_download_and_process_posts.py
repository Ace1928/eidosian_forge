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
def download_and_process_posts(post_ids, st_time):
    subreddit_names = set()
    lines = defaultdict(list)
    for i, (name, l) in enumerate(get_posts(post_ids)):
        if i % 1000000 == 0:
            print('read %d lines, found %d' % (i, sum([len(ls) for ls in lines.values()])), time() - st_time)
        lines[name] += [l.strip()]
        subreddit_names.add(name)
    print('tokenizing and selecting specific posts %2f' % (time() - st_time))
    processed_items = dict([(name, []) for name in subreddit_names])
    key_list = ['id', 'score', 'url', 'title', 'selftext']
    for name in subreddit_names:
        for line in lines[name]:
            reddit_dct = json.loads(line)
            if reddit_dct.get('num_comments', 1) > 0:
                reddit_res = {}
                for k in key_list:
                    if k in ['title', 'selftext']:
                        if reddit_dct[k].lower() in ['[removed]', '[deleted]']:
                            reddit_dct[k] = ''
                        txt, url_list = word_url_tokenize(reddit_dct[k])
                        reddit_res[k] = (' '.join(txt.split()), url_list)
                    else:
                        reddit_res[k] = reddit_dct[k]
                processed_items[name] += [reddit_res]
    print('Total found %d' % len(processed_items), time() - st_time)
    return (subreddit_names, processed_items)