import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging

    XLNet write prediction logic (more complex than Bert's). Write final predictions to the json file and log-odds of
    null if needed.

    Requires utils_squad_evaluate.py
    