from jeepney import DBusAddress, new_signal, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
def test_match_rule_arg_path():
    rule = MatchRule(type='method_call')
    rule.add_arg_condition(0, '/aa/bb/', kind='path')
    assert rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('/aa/bb/',)))
    assert rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('/aa/bb/cc',)))
    assert rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('/aa/',)))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('/aa',)))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo', signature='s', body=('/aa/bb',)))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo', signature='u', body=(12,)))