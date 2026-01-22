import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
Integration tests for uploading and downloading to GCS.

These tests exercise most of the corner cases for upload/download of
files in apitools, via GCS. There are no performance tests here yet.
