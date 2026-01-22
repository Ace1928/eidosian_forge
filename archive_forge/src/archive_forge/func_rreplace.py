import argparse
import os
import torch
from transformers import FlavaImageCodebook, FlavaImageCodebookConfig
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)