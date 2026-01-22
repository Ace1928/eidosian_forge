import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
@patch.object(TextEnvironment, 'generate', side_effect=dummy_generate)
def test_text_environment_max_calls(self, mock_generate):
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools={'DummyTool': DummyTool()}, reward_fn=lambda x: [torch.tensor(1) for _ in x], prompt='I am a prompt!\n')
    env.max_turns = 1
    _, _, _, _, histories = env.run(['test'])
    assert histories[0].text == 'I am a prompt!\n' + 'test' + 1 * '<request><DummyTool>test<call>test<response>'
    env.max_turns = 2
    _, _, _, _, histories = env.run(['test'])
    assert histories[0].text == 'I am a prompt!\n' + 'test' + 2 * '<request><DummyTool>test<call>test<response>'
    env.max_turns = 4
    _, _, _, _, histories = env.run(['test'])
    assert histories[0].text == 'I am a prompt!\n' + 'test' + 4 * '<request><DummyTool>test<call>test<response>'