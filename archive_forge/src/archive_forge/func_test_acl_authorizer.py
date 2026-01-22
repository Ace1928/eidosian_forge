from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def test_acl_authorizer(self):
    ctx = checkers.AuthContext()
    tests = [('no ops, no problem', bakery.ACLAuthorizer(allow_public=True, get_acl=lambda x, y: []), None, [], []), ('identity that does not implement ACLIdentity; user should be denied except for everyone group', bakery.ACLAuthorizer(allow_public=True, get_acl=lambda ctx, op: [bakery.EVERYONE] if op.entity == 'a' else ['alice']), SimplestIdentity('bob'), [bakery.Op(entity='a', action='a'), bakery.Op(entity='b', action='b')], [True, False]), ('identity that does not implement ACLIdentity with user == Id; user should be denied except for everyone group', bakery.ACLAuthorizer(allow_public=True, get_acl=lambda ctx, op: [bakery.EVERYONE] if op.entity == 'a' else ['bob']), SimplestIdentity('bob'), [bakery.Op(entity='a', action='a'), bakery.Op(entity='b', action='b')], [True, False]), ('permission denied for everyone without AllowPublic', bakery.ACLAuthorizer(allow_public=False, get_acl=lambda x, y: [bakery.EVERYONE]), SimplestIdentity('bob'), [bakery.Op(entity='a', action='a')], [False]), ('permission granted to anyone with no identity with AllowPublic', bakery.ACLAuthorizer(allow_public=True, get_acl=lambda x, y: [bakery.EVERYONE]), None, [bakery.Op(entity='a', action='a')], [True])]
    for test in tests:
        allowed, caveats = test[1].authorize(ctx, test[2], test[3])
        self.assertEqual(len(caveats), 0)
        self.assertEqual(allowed, test[4])