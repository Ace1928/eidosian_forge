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
def gather_dump_urls(base_url, mode):
    page = requests.get(base_url + mode)
    soup = BeautifulSoup(page.content, 'lxml')
    files = [it for it in soup.find_all(attrs={'class': 'file'})]
    f_urls = [tg.find_all(lambda x: x.has_attr('href'))[0]['href'] for tg in files if len(tg.find_all(lambda x: x.has_attr('href'))) > 0]
    date_to_url = {}
    for url_st in f_urls:
        ls = re.findall('20[0-9]{2}-[0-9]{2}', url_st)
        if len(ls) > 0:
            yr, mt = ls[0].split('-')
            date_to_url[int(yr), int(mt)] = base_url + mode + url_st[1:]
    return date_to_url