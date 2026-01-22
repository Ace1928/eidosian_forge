from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def postprocess_question_answering(answer: dict) -> tuple[str, dict]:
    return (answer['answer'], {answer['answer']: answer['score']})