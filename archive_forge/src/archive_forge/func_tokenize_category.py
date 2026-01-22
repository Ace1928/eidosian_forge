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
def tokenize_category(self, text, category):
    if category == 'text':
        return self.tokenize_text(text)
    elif category == 'code':
        return self.tokenize_code(text)
    elif category == 'math':
        return self.tokenize_math(text)
    elif category == 'logic':
        return self.tokenize_logic(text)
    else:
        raise ValueError(f'Unknown category: {category}')