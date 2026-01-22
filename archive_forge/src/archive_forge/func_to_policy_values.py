from oslo_context import context as oslo_context
def to_policy_values(self):
    """Add keystone-specific policy values to policy representation.

        This method converts generic policy values to a dictionary form using
        the base implementation from oslo_context.context.RequestContext.
        Afterwards, it is going to pull keystone-specific values off the
        context and represent them as items in the policy values dictionary.
        This is because keystone uses default policies that rely on these
        values, so we need to guarantee they are present during policy
        enforcement if they are present on the context object.

        This method is automatically called in
        oslo_policy.policy.Enforcer.enforce() if oslo.policy knows it's dealing
        with a context object.

        """
    values = super(RequestContext, self).to_policy_values()
    values['token'] = self.token_reference['token']
    values['domain_id'] = self.domain_id if self.domain_id else None
    return values