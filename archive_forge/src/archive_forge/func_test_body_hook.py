import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_body_hook(self):
    test_hook = TestHook('foo')
    client, directive = self.compose_with_hooks([test_hook])
    self.assertEqual(1, len(test_hook.calls))
    self.assertEqual('foo', client.body)
    params = test_hook.calls[0]
    self.assertIsInstance(params, merge_directive.MergeRequestBodyParams)
    self.assertIs(None, params.body)
    self.assertIs(None, params.orig_body)
    self.assertEqual('jrandom@example.com', params.to)
    self.assertEqual('[MERGE] This code rox', params.subject)
    self.assertEqual(directive, params.directive)
    self.assertEqual('foo-1', params.basename)