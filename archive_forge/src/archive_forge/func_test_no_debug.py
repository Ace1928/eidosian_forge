import http.client as http
import os
import stat
import httplib2
from glance.tests import functional
def test_no_debug(self):
    """
        Test logging output proper when debug is off.
        """
    self.cleanup()
    self.start_servers(debug=False)
    self.assertTrue(os.path.exists(self.api_server.log_file))
    with open(self.api_server.log_file, 'r') as f:
        api_log_out = f.read()
    self.assertNotIn('DEBUG glance', api_log_out)
    self.stop_servers()