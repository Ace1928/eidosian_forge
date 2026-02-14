import unittest
from unittest.mock import patch, MagicMock
from download_and_save import fetch_open_source_models, list_available_sizes, list_available_formats

class TestDownloadAndSave(unittest.TestCase):
    @patch('download_and_save.API.list_models')
    @patch('download_and_save.AutoConfig.from_pretrained')
    def test_fetch_open_source_models(self, mock_from_pretrained, mock_list_models):
        mock_list_models.return_value = [
            MagicMock(modelId='model1', private=False),
            MagicMock(modelId='model2', private=True),
            MagicMock(modelId='model3', private=False)
        ]
        mock_from_pretrained.side_effect = [None, Exception('Error'), None]

        models = fetch_open_source_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].modelId, 'model1')
        self.assertEqual(models[1].modelId, 'model3')

    @patch('download_and_save.API.list_repo_files')
    def test_list_available_sizes(self, mock_list_repo_files):
        mock_list_repo_files.return_value = ['file1.bin', 'file2.pt', 'file3.txt']
        sizes = list_available_sizes('model1')
        self.assertEqual(len(sizes), 2)

    @patch('download_and_save.API.list_repo_files')
    def test_list_available_formats(self, mock_list_repo_files):
        mock_list_repo_files.return_value = ['file1.bin', 'file2.pt', 'file3.txt']
        formats = list_available_formats('model1')
        self.assertIn('bin', formats)
        self.assertIn('pt', formats)
        self.assertIn('txt', formats)

if __name__ == '__main__':
    unittest.main() 
