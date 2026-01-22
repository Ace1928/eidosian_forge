import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from transformers import (
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
def read_txt_into_dict(filename):
    result = {}
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file):
            line = line.strip()
            if line:
                words = line.split()
                key = line_number
                value = words[0]
                result[key] = value
    return result