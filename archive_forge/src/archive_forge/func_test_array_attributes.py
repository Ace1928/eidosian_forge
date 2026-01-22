import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
def test_array_attributes(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        dt = h5py.string_dtype('utf-8')
        unicode = 'unicodé'
        ds.attrs['unicode'] = unicode
        ds.attrs['unicode_0dim'] = np.array(unicode, dtype=dt)
        ds.attrs['unicode_1dim'] = np.array([unicode], dtype=dt)
        ds.attrs['unicode_arrary'] = np.array([unicode, 'foobár'], dtype=dt)
        ds.attrs['unicode_list'] = [unicode]
        dt = h5py.string_dtype('ascii')
        ascii = 'ascii'
        ds.attrs['ascii'] = ascii
        ds.attrs['ascii_0dim'] = np.array(ascii, dtype=dt)
        ds.attrs['ascii_1dim'] = np.array([ascii], dtype=dt)
        ds.attrs['ascii_array'] = np.array([ascii, 'foobar'], dtype=dt)
        ds.attrs['ascii_list'] = [ascii]
        ascii = b'ascii'
        ds.attrs['bytes'] = ascii
        ds.attrs['bytes_0dim'] = np.array(ascii, dtype=dt)
        ds.attrs['bytes_1dim'] = np.array([ascii], dtype=dt)
        ds.attrs['bytes_array'] = np.array([ascii, b'foobar'], dtype=dt)
        ds.attrs['bytes_list'] = [ascii]
        dt = h5py.string_dtype('utf-8', 10)
        ds.attrs['unicode_fixed'] = np.array(unicode.encode('utf-8'), dtype=dt)
        ds.attrs['unicode_fixed_0dim'] = np.array(unicode.encode('utf-8'), dtype=dt)
        ds.attrs['unicode_fixed_1dim'] = np.array([unicode.encode('utf-8')], dtype=dt)
        ds.attrs['unicode_fixed_arrary'] = np.array([unicode.encode('utf-8'), 'foobár'.encode()], dtype=dt)
        dt = h5py.string_dtype('ascii', 10)
        ascii = 'ascii'
        ds.attrs['ascii_fixed'] = np.array(ascii, dtype=dt)
        ds.attrs['ascii_fixed_0dim'] = np.array(ascii, dtype=dt)
        ds.attrs['ascii_fixed_1dim'] = np.array([ascii], dtype=dt)
        ds.attrs['ascii_fixed_array'] = np.array([ascii, 'foobar'], dtype=dt)
        ascii = b'ascii'
        ds.attrs['bytes_fixed'] = np.array(ascii, dtype=dt)
        ds.attrs['bytes_fixed_0dim'] = np.array(ascii, dtype=dt)
        ds.attrs['bytes_fixed_1dim'] = np.array([ascii], dtype=dt)
        ds.attrs['bytes_fixed_array'] = np.array([ascii, b'foobar'], dtype=dt)
        ds.attrs['int'] = 1
        ds.attrs['intlist'] = [1]
        ds.attrs['int_array'] = np.arange(10)
        ds.attrs['empty_list'] = []
        ds.attrs['empty_array'] = np.array([])
    with h5netcdf.File(tmp_local_netcdf, mode='r') as ds:
        assert ds.attrs['unicode'] == unicode
        assert ds.attrs['unicode_0dim'] == unicode
        assert ds.attrs['unicode_1dim'] == unicode
        assert ds.attrs['unicode_arrary'] == [unicode, 'foobár']
        assert ds.attrs['unicode_list'] == unicode
        ascii = 'ascii'
        foobar = 'foobar'
        assert ds.attrs['ascii'] == 'ascii'
        assert ds.attrs['ascii_0dim'] == ascii
        assert ds.attrs['ascii_1dim'] == ascii
        assert ds.attrs['ascii_array'] == [ascii, foobar]
        assert ds.attrs['ascii_list'] == 'ascii'
        assert ds.attrs['bytes'] == ascii
        assert ds.attrs['bytes_0dim'] == ascii
        assert ds.attrs['bytes_1dim'] == ascii
        assert ds.attrs['bytes_array'] == [ascii, foobar]
        assert ds.attrs['bytes_list'] == 'ascii'
        assert ds.attrs['unicode_fixed'] == unicode
        assert ds.attrs['unicode_fixed_0dim'] == unicode
        assert ds.attrs['unicode_fixed_1dim'] == unicode
        assert ds.attrs['unicode_fixed_arrary'] == [unicode, 'foobár']
        ascii = 'ascii'
        assert ds.attrs['ascii_fixed'] == ascii
        assert ds.attrs['ascii_fixed_0dim'] == ascii
        assert ds.attrs['ascii_fixed_1dim'] == ascii
        assert ds.attrs['ascii_fixed_array'] == [ascii, 'foobar']
        assert ds.attrs['bytes_fixed'] == ascii
        assert ds.attrs['bytes_fixed_0dim'] == ascii
        assert ds.attrs['bytes_fixed_1dim'] == ascii
        assert ds.attrs['bytes_fixed_array'] == [ascii, 'foobar']
        assert ds.attrs['int'] == 1
        assert ds.attrs['intlist'] == 1
        np.testing.assert_equal(ds.attrs['int_array'], np.arange(10))
        np.testing.assert_equal(ds.attrs['empty_list'], np.array([]))
        np.testing.assert_equal(ds.attrs['empty_array'], np.array([]))
    with legacyapi.Dataset(tmp_local_netcdf, mode='r') as ds:
        assert ds.unicode == unicode
        assert ds.unicode_0dim == unicode
        assert ds.unicode_1dim == unicode
        assert ds.unicode_arrary == [unicode, 'foobár']
        assert ds.unicode_list == unicode
        ascii = 'ascii'
        foobar = 'foobar'
        assert ds.ascii == 'ascii'
        assert ds.ascii_0dim == ascii
        assert ds.ascii_1dim == ascii
        assert ds.ascii_array == [ascii, foobar]
        assert ds.ascii_list == 'ascii'
        assert ds.bytes == ascii
        assert ds.bytes_0dim == ascii
        assert ds.bytes_1dim == ascii
        assert ds.bytes_array == [ascii, foobar]
        assert ds.bytes_list == 'ascii'
        assert ds.unicode_fixed == unicode
        assert ds.unicode_fixed_0dim == unicode
        assert ds.unicode_fixed_1dim == unicode
        assert ds.unicode_fixed_arrary == [unicode, 'foobár']
        ascii = 'ascii'
        assert ds.ascii_fixed == ascii
        assert ds.ascii_fixed_0dim == ascii
        assert ds.ascii_fixed_1dim == ascii
        assert ds.ascii_fixed_array == [ascii, 'foobar']
        assert ds.bytes_fixed == ascii
        assert ds.bytes_fixed_0dim == ascii
        assert ds.bytes_fixed_1dim == ascii
        assert ds.bytes_fixed_array == [ascii, 'foobar']
        assert ds.int == 1
        assert ds.intlist == 1
        np.testing.assert_equal(ds.int_array, np.arange(10))
        np.testing.assert_equal(ds.attrs['empty_list'], np.array([]))
        np.testing.assert_equal(ds.attrs['empty_array'], np.array([]))
    with netCDF4.Dataset(tmp_local_netcdf, mode='r') as ds:
        assert ds.unicode == unicode
        assert ds.unicode_0dim == unicode
        assert ds.unicode_1dim == unicode
        assert ds.unicode_arrary == [unicode, 'foobár']
        assert ds.unicode_list == unicode
        ascii = 'ascii'
        assert ds.ascii == ascii
        assert ds.ascii_0dim == ascii
        assert ds.ascii_1dim == ascii
        assert ds.ascii_array == [ascii, 'foobar']
        assert ds.ascii_list == ascii
        assert ds.bytes == ascii
        assert ds.bytes_0dim == ascii
        assert ds.bytes_1dim == ascii
        assert ds.bytes_array == [ascii, 'foobar']
        assert ds.bytes_list == ascii
        assert ds.unicode_fixed == unicode
        assert ds.unicode_fixed_0dim == unicode
        assert ds.unicode_fixed_1dim == unicode
        assert ds.unicode_fixed_arrary == [unicode, 'foobár']
        assert ds.ascii_fixed == ascii
        assert ds.ascii_fixed_0dim == ascii
        assert ds.ascii_fixed_1dim == ascii
        assert ds.ascii_fixed_array == [ascii, 'foobar']
        assert ds.bytes_fixed == ascii
        assert ds.bytes_fixed_0dim == ascii
        assert ds.bytes_fixed_1dim == ascii
        assert ds.bytes_fixed_array == [ascii, 'foobar']
        assert ds.int == 1
        assert ds.intlist == 1
        np.testing.assert_equal(ds.int_array, np.arange(10))
        np.testing.assert_equal(ds.empty_list, np.array([]))
        np.testing.assert_equal(ds.empty_array, np.array([]))