from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def train_dictionary(dict_size, samples, k=0, d=0, f=0, split_point=0.0, accel=0, notifications=0, dict_id=0, level=0, steps=0, threads=0):
    """Train a dictionary from sample data using the COVER algorithm.

    A compression dictionary of size ``dict_size`` will be created from the
    iterable of ``samples``. The raw dictionary bytes will be returned.

    The dictionary training mechanism is known as *cover*. More details about it
    are available in the paper *Effective Construction of Relative Lempel-Ziv
    Dictionaries* (authors: Liao, Petri, Moffat, Wirth).

    The cover algorithm takes parameters ``k`` and ``d``. These are the
    *segment size* and *dmer size*, respectively. The returned dictionary
    instance created by this function has ``k`` and ``d`` attributes
    containing the values for these parameters. If a ``ZstdCompressionDict``
    is constructed from raw bytes data (a content-only dictionary), the
    ``k`` and ``d`` attributes will be ``0``.

    The segment and dmer size parameters to the cover algorithm can either be
    specified manually or ``train_dictionary()`` can try multiple values
    and pick the best one, where *best* means the smallest compressed data size.
    This later mode is called *optimization* mode.

    Under the hood, this function always calls
    ``ZDICT_optimizeTrainFromBuffer_fastCover()``. See the corresponding C library
    documentation for more.

    If neither ``steps`` nor ``threads`` is defined, defaults for ``d``, ``steps``,
    and ``level`` will be used that are equivalent with what
    ``ZDICT_trainFromBuffer()`` would use.


    :param dict_size:
       Target size in bytes of the dictionary to generate.
    :param samples:
       A list of bytes holding samples the dictionary will be trained from.
    :param k:
       Segment size : constraint: 0 < k : Reasonable range [16, 2048+]
    :param d:
       dmer size : constraint: 0 < d <= k : Reasonable range [6, 16]
    :param f:
       log of size of frequency array : constraint: 0 < f <= 31 : 1 means
       default(20)
    :param split_point:
       Percentage of samples used for training: Only used for optimization.
       The first # samples * ``split_point`` samples will be used to training.
       The last # samples * (1 - split_point) samples will be used for testing.
       0 means default (0.75), 1.0 when all samples are used for both training
       and testing.
    :param accel:
       Acceleration level: constraint: 0 < accel <= 10. Higher means faster
       and less accurate, 0 means default(1).
    :param dict_id:
       Integer dictionary ID for the produced dictionary. Default is 0, which uses
       a random value.
    :param steps:
       Number of steps through ``k`` values to perform when trying parameter
       variations.
    :param threads:
       Number of threads to use when trying parameter variations. Default is 0,
       which means to use a single thread. A negative value can be specified to
       use as many threads as there are detected logical CPUs.
    :param level:
       Integer target compression level when trying parameter variations.
    :param notifications:
       Controls writing of informational messages to ``stderr``. ``0`` (the
       default) means to write nothing. ``1`` writes errors. ``2`` writes
       progression info. ``3`` writes more details. And ``4`` writes all info.
    """
    if not isinstance(samples, list):
        raise TypeError('samples must be a list')
    if threads < 0:
        threads = _cpu_count()
    if not steps and (not threads):
        d = d or 8
        steps = steps or 4
        level = level or 3
    total_size = sum(map(len, samples))
    samples_buffer = new_nonzero('char[]', total_size)
    sample_sizes = new_nonzero('size_t[]', len(samples))
    offset = 0
    for i, sample in enumerate(samples):
        if not isinstance(sample, bytes):
            raise ValueError('samples must be bytes')
        l = len(sample)
        ffi.memmove(samples_buffer + offset, sample, l)
        offset += l
        sample_sizes[i] = l
    dict_data = new_nonzero('char[]', dict_size)
    dparams = ffi.new('ZDICT_fastCover_params_t *')[0]
    dparams.k = k
    dparams.d = d
    dparams.f = f
    dparams.steps = steps
    dparams.nbThreads = threads
    dparams.splitPoint = split_point
    dparams.accel = accel
    dparams.zParams.notificationLevel = notifications
    dparams.zParams.dictID = dict_id
    dparams.zParams.compressionLevel = level
    zresult = lib.ZDICT_optimizeTrainFromBuffer_fastCover(ffi.addressof(dict_data), dict_size, ffi.addressof(samples_buffer), ffi.addressof(sample_sizes, 0), len(samples), ffi.addressof(dparams))
    if lib.ZDICT_isError(zresult):
        msg = ffi.string(lib.ZDICT_getErrorName(zresult)).decode('utf-8')
        raise ZstdError('cannot train dict: %s' % msg)
    return ZstdCompressionDict(ffi.buffer(dict_data, zresult)[:], dict_type=DICT_TYPE_FULLDICT, k=dparams.k, d=dparams.d)