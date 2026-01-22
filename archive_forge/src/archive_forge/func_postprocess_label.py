from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def postprocess_label(scores: list[dict[str, str | float]]) -> dict:
    return {c['label']: c['score'] for c in scores}