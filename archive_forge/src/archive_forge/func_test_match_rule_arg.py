from jeepney import DBusAddress, new_signal, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
def test_match_rule_arg():
    rule = MatchRule(type='method_call')
    rule.add_arg_condition(0, 'foo')
    assert rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('foo',)))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('foobar',)))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo'))