from typing import Optional
import argparse
import os
import asyncio
from glob import glob
import orjson
import openai
from tqdm import tqdm
from openai.error import RateLimitError, ServiceUnavailableError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from vllm import LLM, SamplingParams
from transformers.utils.hub import cached_file
from ochat.evaluation.match_answer import MATCH_ANSWER_FUNCTION
from ochat.config import MODEL_CONFIG_MAP
def tokenize_questions(model_config: object, conv_template: object, questions: list, condition: str, system_msg: str):
    from ochat.config import Conversation, Message
    prompt_indices = []
    conversations = []
    for idx, q in enumerate(questions):
        if q['response']:
            continue
        conversations.append(Conversation(items=[Message(role='user', content=q['question']), Message(role='assistant', content='')], condition=condition, system=system_msg))
        prompt_indices.append(idx)
    conversations, _ = conv_template.tokenize_conversations(conversations, inference=True)
    conversations = [tokens[-model_config.model_max_context:] for tokens in conversations]
    return (conversations, prompt_indices)