import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_generate_v4(self):
    """Test v4 generator with data from AWS docs example.

        see:
        http://docs.aws.amazon.com/general/latest/gr/
        sigv4-create-canonical-request.html
        and
        http://docs.aws.amazon.com/general/latest/gr/
        sigv4-signed-request-examples.html
        """
    secret = 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'
    signer = utils.Ec2Signer(secret)
    body_hash = 'b6359072c78d70ebee1e81adcbab4f01bf2c23245fa365ef83fe8f1f955085e2'
    auth_str = 'AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20110909/us-east-1/iam/aws4_request,SignedHeaders=content-type;host;x-amz-date,'
    headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8', 'X-Amz-Date': '20110909T233600Z', 'Host': 'iam.amazonaws.com', 'Authorization': auth_str}
    params = {'Action': 'CreateUser', 'UserName': 'NewUser', 'Version': '2010-05-08', 'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': 'AKIAEXAMPLE/20140611/us-east-1/iam/aws4_request', 'X-Amz-Date': '20140611T231318Z', 'X-Amz-Expires': '30', 'X-Amz-SignedHeaders': 'host', 'X-Amz-Signature': 'ced6826de92d2bdeed8f846f0bf508e8559e98e4b0199114b84c54174deb456c'}
    credentials = {'host': 'iam.amazonaws.com', 'verb': 'POST', 'path': '/', 'params': params, 'headers': headers, 'body_hash': body_hash}
    signature = signer.generate(credentials)
    expected = 'ced6826de92d2bdeed8f846f0bf508e8559e98e4b0199114b84c54174deb456c'
    self.assertEqual(signature, expected)