from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
@property
def chat_template(self):
    return f"{{% for message in messages %}}{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}{{% endfor %}}{{% if add_generation_prompt %}}{{{{ '{self.assistant}\n' }}}}{{% endif %}}"