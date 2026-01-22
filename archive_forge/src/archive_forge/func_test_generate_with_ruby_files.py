import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_ruby_files(self):
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    self.write_file('config.ru', 'run Index.app')
    unstub = self.stub_response('bundle exec rackup -p $PORT -E deployment')
    self.generate_configs()
    unstub()
    app_yaml = self.file_contents('app.yaml')
    self.assertIn('runtime: ruby\n', app_yaml)
    self.assertIn('env: flex\n', app_yaml)
    self.assertIn('entrypoint: bundle exec rackup -p $PORT -E deployment\n', app_yaml)
    self.assertFalse(os.path.exists(self.full_path('Dockerfile')))
    self.assertFalse(os.path.exists(self.full_path('.dockerignore')))