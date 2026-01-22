import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_ruby_files_no_write(self):
    """Tests generate_config_data with basic Ruby files.

        Tests that app.yaml is written with correct contents given entrypoint
        response, and that Dockerfile and .dockerignore not written to disk.
        """
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    self.write_file('config.ru', 'run Index.app')
    unstub = self.stub_response('bundle exec rackup -p $PORT -E deployment')
    cfg_files = self.generate_config_data()
    unstub()
    app_yaml = self.file_contents('app.yaml')
    self.assertIn('runtime: ruby\n', app_yaml)
    self.assertIn('env: flex\n', app_yaml)
    self.assertIn('entrypoint: bundle exec rackup -p $PORT -E deployment\n', app_yaml)
    self.assertNotIn('Dockerfile', [f.filename for f in cfg_files])
    self.assertNotIn('.dockerignore', [f.filename for f in cfg_files])