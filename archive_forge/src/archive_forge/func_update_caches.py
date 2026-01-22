import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import os
import json
from collections import defaultdict, deque
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math
import tensorboard
from torch.utils.tensorboard import SummaryWriter
def update_caches(self, tokens, category):
    cache = self.category_caches[category]
    for token in tokens:
        if token in cache:
            cache.remove(token)
        cache.append(token)
        if len(cache) > CACHE_SIZE:
            evicted_token = cache.popleft()
            self.category_file_caches[category].add(evicted_token)
    if len(self.category_file_caches[category]) >= FILE_CACHE_THRESHOLD:
        self._dump_file_cache(category)