import asyncio
import cProfile
import hashlib
import io
import itertools
import json
import logging
import resource
import os
import pstats
import queue
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from functools import reduce
from logging.handlers import MemoryHandler, RotatingFileHandler
from logging import StreamHandler, FileHandler
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Callable
import coloredlogs
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import (
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools
def process_logs(self):
    """
        Processes log messages from the queue, either synchronously or asynchronously, capturing detailed profiling, traceback information, and system state.
        This method ensures that it can be called in any context by dynamically managing the event loop and wrapping the asynchronous logic inside a synchronous method if needed.
        """

    async def async_process_logs():
        """
            Continuously processes log messages from the queue in a dedicated thread, capturing detailed profiling, traceback information, and system state.
            """
        while True:
            try:
                record = await self.log_queue.get()
                if record is None:
                    break
                if not self.profiler.is_running():
                    self.profiler.enable()
                self.logger.handle(record)
                self.profiler.disable()
                s = io.StringIO()
                sortby = pstats.SortKey.CUMULATIVE
                ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
                ps.print_stats()
                logging.debug('Profile data:\n%s' % s.getvalue())
                system_state = {'timestamp': datetime.now().isoformat(), 'system_load': os.getloadavg(), 'memory_usage': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                logging.debug(f'System state at log time: {system_state}')
            except Exception as e:
                exc_info = sys.exc_info()
                traceback_details = {'filename': exc_info[2].tb_frame.f_code.co_filename, 'lineno': exc_info[2].tb_lineno, 'name': exc_info[2].tb_frame.f_code.co_name, 'type': exc_info[0].__name__, 'message': str(e)}
                log_msg = 'Logging thread encountered an exception: {details}'.format(details=traceback_details)
                self.logger.critical(log_msg, exc_info=True)
                break
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        asyncio.create_task(async_process_logs())
    else:
        loop.run_until_complete(async_process_logs())