import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging
def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join((ch for ch in text if ch not in exclude))