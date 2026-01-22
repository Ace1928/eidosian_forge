import os
from logging import getLogger
from pathlib import Path
from typing import (
import tiktoken
from tiktoken.load import load_tiktoken_bpe
def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
    tokens = []
    tokens.append(self.tokenizer.special_tokens['<|begin_of_text|>'])
    for message in dialog:
        tokens.extend(self.encode_message(message))
    tokens.extend(self.encode_header({'role': 'assistant', 'content': ''}))
    return tokens