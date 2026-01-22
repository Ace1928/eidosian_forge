import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
@patch.object(TextEnvironment, 'generate', side_effect=dummy_generate)
def test_text_environment_run(self, mock_generate):
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools={'DummyTool': DummyTool()}, reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)], prompt='I am a prompt!\n', max_turns=2)
    task_1 = 'Hello there!'
    task_2 = 'Hello there! General Kenobi!'
    query, response, response_mask, reward, histories = env.run([task_1, task_2])
    assert len(query[0]) == 9
    assert len(query[1]) == 12
    assert len(response[0]) == 14
    assert len(response[1]) == 14
    assert response_mask[0].sum() == 2 * 3
    assert response_mask[1].sum() == 2 * 3
    assert reward[0] == 0
    assert reward[1] == 1
    assert histories[0].text == 'I am a prompt!\n' + 'Hello there!' + 2 * '<request><DummyTool>test<call>test<response>'
    assert histories[1].text == 'I am a prompt!\n' + 'Hello there! General Kenobi!' + 2 * '<request><DummyTool>test<call>test<response>'