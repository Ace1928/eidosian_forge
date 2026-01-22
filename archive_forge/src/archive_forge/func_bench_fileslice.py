import sys
from io import BytesIO
from timeit import timeit
import numpy as np
from ..fileslice import fileslice
from ..openers import ImageOpener
from ..optpkg import optional_package
from ..rstutils import rst_table
from ..tmpdirs import InTemporaryDirectory
def bench_fileslice(bytes=True, file_=True, gz=True, bz2=False, zst=True):
    sys.stdout.flush()
    repeat = 2

    def my_table(title, times, base):
        print()
        print(rst_table(times, ROW_NAMES, COL_NAMES, title, val_fmt='{0[0]:3.2f} ({0[1]:3.2f})'))
        print(f'Base time: {base:3.2f}')
    if bytes:
        fobj = BytesIO()
        times, base = run_slices(fobj, repeat)
        my_table('Bytes slice - raw (ratio)', np.dstack((times, times / base)), base)
    if file_:
        with InTemporaryDirectory():
            file_times, file_base = run_slices('data.bin', repeat)
        my_table('File slice - raw (ratio)', np.dstack((file_times, file_times / file_base)), file_base)
    if gz:
        with InTemporaryDirectory():
            gz_times, gz_base = run_slices('data.gz', repeat)
        my_table('gz slice - raw (ratio)', np.dstack((gz_times, gz_times / gz_base)), gz_base)
    if bz2:
        with InTemporaryDirectory():
            bz2_times, bz2_base = run_slices('data.bz2', repeat)
        my_table('bz2 slice - raw (ratio)', np.dstack((bz2_times, bz2_times / bz2_base)), bz2_base)
    if zst and HAVE_ZSTD:
        with InTemporaryDirectory():
            zst_times, zst_base = run_slices('data.zst', repeat)
        my_table('zst slice - raw (ratio)', np.dstack((zst_times, zst_times / zst_base)), zst_base)
    sys.stdout.flush()