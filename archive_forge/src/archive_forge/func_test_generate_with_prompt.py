import os
import unittest
from gae_ext_runtime import testutil
def test_generate_with_prompt(self):
    self.write_file('index.rb', 'class Index; end')
    self.write_file('Gemfile', 'source "https://rubygems.org"')
    unstub = self.stub_response('bundle exec ruby index.rb $PORT')
    self.generate_configs(deploy=True)
    unstub()
    dockerfile = self.file_contents('Dockerfile')
    self.assertEqual(dockerfile, DOCKERFILE_TEXT.format(ruby_version='', entrypoint='bundle exec ruby index.rb $PORT'))
    dockerignore = self.file_contents('.dockerignore')
    self.assertIn('.dockerignore\n', dockerignore)
    self.assertIn('Dockerfile\n', dockerignore)
    self.assertIn('.git\n', dockerignore)
    self.assertIn('.hg\n', dockerignore)
    self.assertIn('.svn\n', dockerignore)