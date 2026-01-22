import unittest
from boto.s3.cors import CORSConfiguration
def test_two_rules(self):
    cfg = CORSConfiguration()
    cfg.add_rule(['PUT', 'POST', 'DELETE'], 'http://www.example.com', allowed_header='*', max_age_seconds=3000, expose_header='x-amz-server-side-encryption')
    cfg.add_rule('GET', '*', allowed_header='*', max_age_seconds=3000)
    self.assertEqual(cfg.to_xml(), CORS_BODY_2)