import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def pad_to_4(x):
    return x + [None] * (4 - len(x))