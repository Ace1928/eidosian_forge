import re
import gc
from time import sleep
from lxml import html
from collections import OrderedDict
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from .browser import get_chrome
def translate_html_slow(code, lf, lt):
    code = ' '.join(code.split())
    regular = re.compile('(?<=[>]).*?(?=[<])', re.S)
    text_parts = regular.findall(code)
    text_parts = [x.strip() for x in text_parts if len(x.strip()) > 3]
    qparts = Queue()
    for p in text_parts:
        qparts.put(p)
    result = OrderedDict()
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(worker, qparts, lf, lt, result)
    for part in result:
        code = code.replace(part, result[part], 1)
    gc.collect()
    return code