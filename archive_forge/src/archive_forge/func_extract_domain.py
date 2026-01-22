import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
def extract_domain(url):
    """Extract domain name from URL string"""
    o = urlparse(url)
    if o.scheme == '' and o.netloc == '':
        o = urlparse('//' + url.lstrip('/'))
    return o.netloc