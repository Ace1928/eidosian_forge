from jeepney import DBusAddress, new_signal, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
def test_match_rule_path_namespace():
    assert MatchRule(path_namespace='/org/freedesktop/portal').matches(new_signal(portal_req_iface, 'Response'))
    assert not MatchRule(path_namespace='/org/freedesktop/por').matches(new_signal(portal_req_iface, 'Response'))