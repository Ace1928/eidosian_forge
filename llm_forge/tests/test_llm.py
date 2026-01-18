import unittest
from unittest.mock import MagicMock, patch
from eidosian_forge.llm_forge import LLMForge

class TestLLMForge(unittest.TestCase):
    def setUp(self):
        self.forge = LLMForge()

    @patch('requests.post')
    def test_generate_with_options(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"key": "val"}'}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        result = self.forge.generate("JSON prompt", options={"temperature": 0}, json_mode=True)
        self.assertTrue(result["success"])
        # Verify JSON format was passed in request
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['format'], 'json')
        self.assertEqual(kwargs['json']['options']['temperature'], 0)

    @patch('requests.get')
    def test_list_local_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        models = self.forge.list_local_models()
        self.assertIn("llama3", models)
        self.assertIn("mistral", models)

if __name__ == "__main__":
    unittest.main()
