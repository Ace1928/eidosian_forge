import copy
import os
import pickle
import warnings
import numpy as np
def writeHDF5(self, fileName, **opts):
    comp = self.defaultCompression
    if isinstance(comp, tuple):
        comp, copts = comp
    else:
        copts = None
    dsOpts = {'compression': comp, 'chunks': True}
    if copts is not None:
        dsOpts['compression_opts'] = copts
    appAxis = opts.get('appendAxis', None)
    if appAxis is not None:
        appAxis = self._interpretAxis(appAxis)
        cs = [min(100000, x) for x in self.shape]
        cs[appAxis] = 1
        dsOpts['chunks'] = tuple(cs)
    else:
        cs = [min(100000, x) for x in self.shape]
        for i in range(self.ndim):
            if 'cols' in self._info[i]:
                cs[i] = 1
        dsOpts['chunks'] = tuple(cs)
    for k in dsOpts:
        if k in opts:
            dsOpts[k] = opts[k]
    if opts.get('mappable', False):
        dsOpts = {'chunks': None, 'compression': None}
    append = False
    if appAxis is not None:
        maxShape = list(self.shape)
        ax = self._interpretAxis(appAxis)
        maxShape[ax] = None
        if os.path.exists(fileName):
            append = True
        dsOpts['maxshape'] = tuple(maxShape)
    else:
        dsOpts['maxshape'] = None
    if append:
        f = h5py.File(fileName, 'r+')
        if f.attrs['MetaArray'] != MetaArray.version:
            raise Exception('The file %s was created with a different version of MetaArray. Will not modify.' % fileName)
        data = f['data']
        shape = list(data.shape)
        shape[ax] += self.shape[ax]
        data.resize(tuple(shape))
        sl = [slice(None)] * len(data.shape)
        sl[ax] = slice(-self.shape[ax], None)
        data[tuple(sl)] = self.view(np.ndarray)
        axKeys = ['values']
        axKeys.extend(opts.get('appendKeys', []))
        axInfo = f['info'][str(ax)]
        for key in axKeys:
            if key in axInfo:
                v = axInfo[key]
                v2 = self._info[ax][key]
                shape = list(v.shape)
                shape[0] += v2.shape[0]
                v.resize(shape)
                v[-v2.shape[0]:] = v2
            else:
                raise TypeError('Cannot append to axis info key "%s"; this key is not present in the target file.' % key)
        f.close()
    else:
        f = h5py.File(fileName, 'w')
        f.attrs['MetaArray'] = MetaArray.version
        f.create_dataset('data', data=self.view(np.ndarray), **dsOpts)
        if isinstance(dsOpts['chunks'], tuple):
            dsOpts['chunks'] = True
            if 'maxshape' in dsOpts:
                del dsOpts['maxshape']
        self.writeHDF5Meta(f, 'info', self._info, **dsOpts)
        f.close()