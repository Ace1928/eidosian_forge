import copy
import os
import pickle
import warnings
import numpy as np
@staticmethod
def readHDF5Meta(root, mmap=False):
    data = {}
    for k in root.attrs:
        val = root.attrs[k]
        if isinstance(val, bytes):
            val = val.decode()
        if isinstance(val, str):
            try:
                val = eval(val)
            except:
                raise Exception('Can not evaluate string: "%s"' % val)
        data[k] = val
    for k in root:
        obj = root[k]
        if isinstance(obj, h5py.Group):
            val = MetaArray.readHDF5Meta(obj)
        elif isinstance(obj, h5py.Dataset):
            if mmap:
                val = MetaArray.mapHDF5Array(obj)
            else:
                val = obj[:]
        else:
            raise Exception("Don't know what to do with type '%s'" % str(type(obj)))
        data[k] = val
    typ = root.attrs['_metaType_']
    try:
        typ = typ.decode('utf-8')
    except:
        pass
    del data['_metaType_']
    if typ == 'dict':
        return data
    elif typ == 'list' or typ == 'tuple':
        d2 = [None] * len(data)
        for k in data:
            d2[int(k)] = data[k]
        if typ == 'tuple':
            d2 = tuple(d2)
        return d2
    else:
        raise Exception("Don't understand metaType '%s'" % typ)