import ray
from ochat.config import Message, Conversation
def get_eot_tokens(self):
    assert len(self.conv_template.eot_tokens_) == 1
    return self.conv_template.eot_tokens_