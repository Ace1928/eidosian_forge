import re
def build_port_bindings(ports):
    port_bindings = {}
    for port in ports:
        internal_port_range, external_range = split_port(port)
        add_port(port_bindings, internal_port_range, external_range)
    return port_bindings