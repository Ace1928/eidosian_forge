import unittest
from chatmock.config import BASE_INSTRUCTIONS
from chatmock.utils import convert_chat_messages_to_responses_input

class TestChatMockBasic(unittest.TestCase):
    def test_base_instructions_loaded(self):
        """Test that base instructions are loaded from prompt.md."""
        self.assertIsInstance(BASE_INSTRUCTIONS, str)
        self.assertTrue(len(BASE_INSTRUCTIONS) > 0)

    def test_message_conversion(self):
        """Test conversion of chat messages to response input format."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        converted = convert_chat_messages_to_responses_input(messages)
        self.assertIsInstance(converted, list)
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"][0]["text"], "Hello")

if __name__ == '__main__':
    unittest.main()
