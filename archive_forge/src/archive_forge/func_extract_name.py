def extract_name(inst):
    assert inst.opname == 'STORE_FAST' or inst.opname == 'STORE_NAME'
    return inst.argval