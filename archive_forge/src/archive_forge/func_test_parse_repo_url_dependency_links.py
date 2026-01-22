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
def test_parse_repo_url_dependency_links(self):
    result = packaging.parse_dependency_links([self.requirements])
    self.assertEqual(['git+git://git.pro-ject.org/oslo.messaging#egg=oslo.messaging-1.0.0-rc', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize-beta', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize2-beta', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize2-beta-4.0.1', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize2-beta-1.0.0-alpha.beta.1', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize2-beta-1.0.0-alpha-a.b-c-somethinglong+build.1-aef.1-its-okay', 'git+git://git.pro-ject.org/django-thumborize#egg=django-thumborize2-beta-2.0.0-rc.1+build.123', 'git+git://git.project.org/Proj#egg=Proj1', 'git+https://git.project.org/Proj#egg=Proj2-0.0.1', 'git+ssh://git.project.org/Proj#egg=Proj3', 'svn+svn://svn.project.org/svn/Proj#egg=Proj4-0.0.2', 'svn+http://svn.project.org/svn/Proj/trunk@2019#egg=Proj5', 'hg+http://hg.project.org/Proj@da39a3ee5e6b#egg=Proj-0.0.3', 'hg+http://hg.project.org/Proj@2019#egg=Proj', 'hg+http://hg.project.org/Proj@v1.0#egg=Proj-0.0.4', 'hg+http://hg.project.org/Proj@special_feature#egg=Proj', 'git://foo.com/zipball#egg=foo-bar-1.2.4'], result)