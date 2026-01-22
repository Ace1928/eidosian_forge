import os
def test_live_application(http_request):
    response = http_request(method='GET', url=TEST_APP_URL)
    assert response.status == 200, response.data.decode('utf-8')