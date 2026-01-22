import textwrap
from oslotest import base
from oslo_policy import policy
from oslo_policy import sphinxext
def test_with_scope_types(self):
    operations = [{'method': 'GET', 'path': '/foo'}, {'method': 'POST', 'path': '/some'}]
    scope_types = ['bar']
    rule = policy.DocumentedRuleDefault('rule_a', '@', 'My sample rule', operations, scope_types=scope_types)
    results = '\n'.join(list(sphinxext._format_policy_section('foo', [rule])))
    self.assertEqual(textwrap.dedent('\n        foo\n        ===\n\n        ``rule_a``\n            :Default: ``@``\n            :Operations:\n                - **GET** ``/foo``\n                - **POST** ``/some``\n            :Scope Types:\n                - **bar**\n\n            My sample rule\n        ').lstrip(), results)