from __future__ import print_function
import os
import fixtures
from pbr import git
from pbr import options
from pbr.tests import base
def test_generate_authors(self):
    author_old = u'Foo Foo <email@foo.com>'
    author_new = u'Bar Bar <email@bar.com>'
    co_author = u'Foo Bar <foo@bar.com>'
    co_author_by = u'Co-authored-by: ' + co_author
    git_log_cmd = 'git --git-dir=%s log --format=%%aN <%%aE>' % self.git_dir
    git_co_log_cmd = 'git --git-dir=%s log' % self.git_dir
    git_top_level = 'git rev-parse --show-toplevel'
    cmd_map = {git_log_cmd: author_new, git_co_log_cmd: co_author_by, git_top_level: self.root_dir}
    exist_files = [self.git_dir, os.path.join(self.temp_path, 'AUTHORS.in')]
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda path: os.path.abspath(path) in exist_files))

    def _fake_run_shell_command(cmd, **kwargs):
        return cmd_map[' '.join(cmd)]
    self.useFixture(fixtures.MonkeyPatch('pbr.git._run_shell_command', _fake_run_shell_command))
    with open(os.path.join(self.temp_path, 'AUTHORS.in'), 'w') as auth_fh:
        auth_fh.write('%s\n' % author_old)
    git.generate_authors(git_dir=self.git_dir, dest_dir=self.temp_path)
    with open(os.path.join(self.temp_path, 'AUTHORS'), 'r') as auth_fh:
        authors = auth_fh.read()
        self.assertIn(author_old, authors)
        self.assertIn(author_new, authors)
        self.assertIn(co_author, authors)