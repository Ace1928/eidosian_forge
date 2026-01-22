import math
from collections.abc import Sequence
import heapq
import json
import torch
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

        Build representation of query, e.g. words or n-grams.

        :param query: string to represent.

        :returns: dictionary containing 'words' dictionary (token => frequency)
                  and 'norm' float (square root of the number of tokens)
        