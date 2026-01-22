import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test__set_project_metadata(self):
    self.assertEqual(len(self.driver._set_project_metadata(None, False, '')), 0)
    md = self.driver._set_project_metadata(None, False, 'this is a test')
    self.assertEqual(len(md), 1)
    self.assertEqual(md[0]['key'], 'sshKeys')
    self.assertEqual(md[0]['value'], 'this is a test')
    md = self.driver._set_project_metadata(None, True, 'this is a test')
    self.assertEqual(len(md), 0)
    gce_md = {'items': [{'key': 'foo', 'value': 'one'}, {'key': 'sshKeys', 'value': 'another test'}]}
    md = self.driver._set_project_metadata(gce_md, False, 'this is a test')
    self.assertEqual(len(md), 2, str(md))
    sshKeys = ''
    count = 0
    for d in md:
        if d['key'] == 'sshKeys':
            count += 1
            sshKeys = d['value']
    self.assertEqual(sshKeys, 'this is a test')
    self.assertEqual(count, 1)
    gce_md = {'items': [{'key': 'foo', 'value': 'one'}, {'key': 'sshKeys', 'value': 'another test'}]}
    md = self.driver._set_project_metadata(gce_md, True, 'this is a test')
    self.assertEqual(len(md), 2, str(md))
    sshKeys = ''
    count = 0
    for d in md:
        if d['key'] == 'sshKeys':
            count += 1
            sshKeys = d['value']
    self.assertEqual(sshKeys, 'another test')
    self.assertEqual(count, 1)
    gce_md = {'items': [{'key': 'foo', 'value': 'one'}, {'key': 'nokeys', 'value': 'two'}]}
    md = self.driver._set_project_metadata(gce_md, True, 'this is a test')
    self.assertEqual(len(md), 2, str(md))
    sshKeys = ''
    count = 0
    for d in md:
        if d['key'] == 'sshKeys':
            count += 1
            sshKeys = d['value']
    self.assertEqual(sshKeys, '')
    self.assertEqual(count, 0)