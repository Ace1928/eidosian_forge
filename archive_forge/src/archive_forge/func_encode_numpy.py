def encode_numpy(obj, chain=None):
    """
    Data encoder for serializing numpy data types.
    """
    if not has_numpy:
        return obj if chain is None else chain(obj)
    if has_cupy and isinstance(obj, cupy.ndarray):
        obj = obj.get()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == 'V':
            kind = b'V'
            descr = obj.dtype.descr
        else:
            kind = b''
            descr = obj.dtype.str
        return {b'nd': True, b'type': descr, b'kind': kind, b'shape': obj.shape, b'data': obj.data if obj.flags['C_CONTIGUOUS'] else obj.tobytes()}
    elif isinstance(obj, (np.bool_, np.number)):
        return {b'nd': False, b'type': obj.dtype.str, b'data': obj.data}
    elif isinstance(obj, complex):
        return {b'complex': True, b'data': obj.__repr__()}
    else:
        return obj if chain is None else chain(obj)