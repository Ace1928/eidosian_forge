import csv
import json
import time
import random
import threading
import numpy as np
import requests
import transformers
import torch
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
def generate_random_prompt(num_tokens):
    generated_tokens_count = 0
    selected_tokens = ''
    while generated_tokens_count < num_tokens:
        selected_tokens += random.choice(vocab)
        selected_tokens += ' '
        generated_tokens_count = len(tokenizer.encode(selected_tokens))
    return selected_tokens