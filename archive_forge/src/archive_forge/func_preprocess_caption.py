import argparse
import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import (
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith('.'):
        return result
    return result + '.'