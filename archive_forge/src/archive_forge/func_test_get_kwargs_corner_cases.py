import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
def test_get_kwargs_corner_cases(self):

    def get_kwargs(tag):
        git_dir = self.repo._basedir + '/.git'
        return packaging._get_increment_kwargs(git_dir, tag)

    def _check_combinations(tag):
        self.repo.commit()
        self.assertEqual(dict(), get_kwargs(tag))
        self.repo.commit('sem-ver: bugfix')
        self.assertEqual(dict(), get_kwargs(tag))
        self.repo.commit('sem-ver: feature')
        self.assertEqual(dict(minor=True), get_kwargs(tag))
        self.repo.uncommit()
        self.repo.commit('sem-ver: deprecation')
        self.assertEqual(dict(minor=True), get_kwargs(tag))
        self.repo.uncommit()
        self.repo.commit('sem-ver: api-break')
        self.assertEqual(dict(major=True), get_kwargs(tag))
        self.repo.commit('sem-ver: deprecation')
        self.assertEqual(dict(major=True, minor=True), get_kwargs(tag))
    _check_combinations('')
    self.repo.tag('1.2.3')
    _check_combinations('1.2.3')