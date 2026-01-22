import textwrap
from oslotest import base
from oslo_policy import policy
from oslo_policy import sphinxext
def test_with_description(self):
    results = '\n'.join(list(sphinxext._format_policy_section('foo', [policy.RuleDefault('rule_a', '@', 'My sample rule')])))
    self.assertEqual(textwrap.dedent('\n        foo\n        ===\n\n        ``rule_a``\n            :Default: ``@``\n\n            My sample rule\n        ').lstrip(), results)