import sys
import unittest
import platform
import pygame
def test_PgObject_GetBuffer_array_interface(self):
    from pygame.bufferproxy import BufferProxy

    class Exporter(self.ExporterBase):

        def get__array_interface__(self):
            return {'version': 3, 'typestr': self.typestr, 'shape': self.shape, 'strides': self.strides, 'data': self.data}
        __array_interface__ = property(get__array_interface__)
        __array_struct__ = property(lambda self: None)
    _shape = [2, 3, 5, 7, 11]
    for ndim in range(1, len(_shape)):
        o = Exporter(_shape[0:ndim], 'i', 2)
        v = BufferProxy(o)
        self.assertSame(v, o)
    ndim = 2
    shape = _shape[0:ndim]
    for typechar in ('i', 'u'):
        for itemsize in (1, 2, 4, 8):
            o = Exporter(shape, typechar, itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)
    for itemsize in (4, 8):
        o = Exporter(shape, 'f', itemsize)
        v = BufferProxy(o)
        self.assertSame(v, o)
    import weakref, gc

    class NoDictError(RuntimeError):
        pass

    class WRDict(dict):
        """Weak referenceable dict"""
        pass

    class Exporter2(Exporter):

        def get__array_interface__2(self):
            self.d = WRDict(Exporter.get__array_interface__(self))
            self.dict_ref = weakref.ref(self.d)
            return self.d
        __array_interface__ = property(get__array_interface__2)

        def free_dict(self):
            self.d = None

        def is_dict_alive(self):
            try:
                return self.dict_ref() is not None
            except AttributeError:
                raise NoDictError('__array_interface__ is unread')
    o = Exporter2((2, 4), 'u', 4)
    v = BufferProxy(o)
    self.assertRaises(NoDictError, o.is_dict_alive)
    length = v.length
    self.assertTrue(o.is_dict_alive())
    o.free_dict()
    gc.collect()
    self.assertFalse(o.is_dict_alive())