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
def process_data(self):
    """
        Initiates the processing of the selected file using AdvancedJSONProcessor.
        """
    if self.file_path:
        try:
            self.json_processor.initialize_processor()
            processed_data = self.json_processor.process_data()
            self.data_manager.store_data(processed_data, 'processed_data.pkl')
            self.operation_status.set('Data processing completed successfully.')
            advanced_logger.log(logging.INFO, f'Data processing completed successfully for {self.file_path}')
        except Exception as e:
            self.operation_status.set(f'Error during data processing: {str(e)}')
            advanced_logger.log(logging.ERROR, f'Data processing failed for {self.file_path}: {e}')
            messagebox.showerror('Processing Error', f'An error occurred: {e}')
    else:
        self.operation_status.set('No file selected for processing.')
        advanced_logger.log(logging.INFO, 'Data processing attempted without a file selected.')