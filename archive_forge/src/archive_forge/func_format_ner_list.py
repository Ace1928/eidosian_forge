from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def format_ner_list(input_string: str, ner_groups: list[dict[str, str | int]]):
    if len(ner_groups) == 0:
        return [(input_string, None)]
    output = []
    end = 0
    prev_end = 0
    for group in ner_groups:
        entity, start, end = (group['entity_group'], group['start'], group['end'])
        output.append((input_string[prev_end:start], None))
        output.append((input_string[start:end], entity))
        prev_end = end
    output.append((input_string[end:], None))
    return output