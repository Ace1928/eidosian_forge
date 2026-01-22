from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import (
def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-if', '--infile', type=str)
    parser.add_argument('-of', '--outfile', type=str)
    parser.add_argument('-histsz', '--history-size', type=int, default=-1, help='The number of turns to include in the prompt.')
    parser.add_argument('-pos', '--positives', type=str, default='positive', help='A comma-separated list of ratings with positive label')
    parser.add_argument('-neg', '--negatives', type=str, default='negative', help='A comma-separated list of ratings with negative label')
    opt = vars(parser.parse_args())
    return opt