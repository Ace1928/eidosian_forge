import os
import unittest
import certifi
def test_read_contents(self) -> None:
    content = certifi.contents()
    assert '-----BEGIN CERTIFICATE-----' in content