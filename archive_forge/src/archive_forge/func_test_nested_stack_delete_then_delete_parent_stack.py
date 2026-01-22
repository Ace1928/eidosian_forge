import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_nested_stack_delete_then_delete_parent_stack(self):
    """Check the robustness of stack deletion.

        This tests that if you manually delete a nested
        stack, the parent stack is still deletable.
        """
    stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ, enable_cleanup=False)
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'secret1')
    self._stack_delete(nested_ident)
    self._stack_delete(stack_identifier)