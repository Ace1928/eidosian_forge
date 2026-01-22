import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import GKE_PARAMS, GKE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.test.container import TestCaseMixin
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.gke import API_VERSION, GKEContainerDriver
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp

    Google Compute Engine Test Class.
    