from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def rows_to_cols(incoming_data: dict) -> dict[str, dict[str, dict[str, list[str]]]]:
    data_column_wise = {}
    for i, header in enumerate(incoming_data['headers']):
        data_column_wise[header] = [str(row[i]) for row in incoming_data['data']]
    return {'inputs': {'data': data_column_wise}}