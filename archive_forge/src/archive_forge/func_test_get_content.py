from unittest import mock
from mistralclient.api.v2 import code_sources
from mistralclient.commands.v2 import code_sources as code_src_cmd
from mistralclient.tests.unit import base
from mistral_lib import actions
def test_get_content(self):
    self.client.code_sources.get.return_value = CODE_SRC
    self.call(code_src_cmd.GetContent, app_args=['hello_module'])
    self.app.stdout.write.assert_called_with(CODE_SRC_CONTENT)