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
def valid_comment(a):
    res = len(a['body'].split()) > 2 and (not a['body'].startswith('Your submission has been removed')) and (a['author'] != 'AutoModerator') and (a['parent_id'] == a['link_id'])
    return res