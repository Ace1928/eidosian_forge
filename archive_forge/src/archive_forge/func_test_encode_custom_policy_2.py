import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_encode_custom_policy_2(self):
    """
        Test base64 encoding custom policy 2 from Amazon's documentation.
        """
    expected = 'eyAKICAgIlN0YXRlbWVudCI6IFt7IAogICAgICAiUmVzb3VyY2UiOiJodHRwOi8vKiIsIAogICAgICAiQ29uZGl0aW9uIjp7IAogICAgICAgICAiSXBBZGRyZXNzIjp7IkFXUzpTb3VyY2VJcCI6IjIxNi45OC4zNS4xLzMyIn0sCiAgICAgICAgICJEYXRlR3JlYXRlclRoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTI0MTA3Mzc5MH0sCiAgICAgICAgICJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTI1NTY3NDcxNn0KICAgICAgfSAKICAgfV0gCn0K'
    encoded = self.dist._url_base64_encode(self.custom_policy_2)
    self.assertEqual(expected, encoded)