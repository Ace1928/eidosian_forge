def str_to_port_no(port_no_str):
    assert len(port_no_str) == _PORT_NO_LEN
    return int(port_no_str, 16)