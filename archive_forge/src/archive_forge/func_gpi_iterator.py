import copy
def gpi_iterator(handle):
    """Read GPI format files.

    This function should be called to read a
    gp_information.goa_uniprot file. At the moment, there is
    only one format, but this may change, so
    this function is a placeholder a future wrapper.
    """
    inline = handle.readline()
    if inline.strip() == '!gpi-version: 1.2':
        return _gpi12iterator(handle)
    elif inline.strip() == '!gpi-version: 1.1':
        return _gpi11iterator(handle)
    elif inline.strip() == '!gpi-version: 1.0':
        return _gpi10iterator(handle)
    elif inline.strip() == '!gpi-version: 2.1':
        raise NotImplementedError('Sorry, parsing GPI version 2 not implemented yet.')
    else:
        raise ValueError(f'Unknown GPI version {inline}\n')