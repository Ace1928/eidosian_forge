import crcmod
def mkPredefinedCrcFun(crc_name):
    definition = _get_definition_by_name(crc_name)
    return crcmod.mkCrcFun(poly=definition['poly'], initCrc=definition['init'], rev=definition['reverse'], xorOut=definition['xor_out'])