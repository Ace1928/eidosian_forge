import copy
from os.path import basename, dirname
from os.path import join as pjoin
import numpy as np
from .. import (
def check_img(img_path, img_klass, sniff_mode, sniff, expect_success, msg):
    """Embedded function to do the actual checks expected."""
    if sniff_mode == 'no_sniff':
        is_img, new_sniff = img_klass.path_maybe_image(img_path)
    elif sniff_mode in ('empty', 'irrelevant', 'bad_sniff'):
        is_img, new_sniff = img_klass.path_maybe_image(img_path, (sniff, img_path))
    else:
        is_img, new_sniff = img_klass.path_maybe_image(img_path, sniff)
    if expect_success:
        new_msg = f'{img_klass.__name__} returned sniff==None ({msg})'
        expected_sizeof_hdr = getattr(img_klass.header_class, 'sizeof_hdr', 0)
        current_sizeof_hdr = 0 if new_sniff is None else len(new_sniff[0])
        assert current_sizeof_hdr >= expected_sizeof_hdr, new_msg
        new_msg = f'{basename(img_path)} ({msg}) image is{('' if is_img else ' not')} a {img_klass.__name__} image.'
        assert is_img, new_msg
    if sniff_mode == 'vanilla':
        return new_sniff
    else:
        return sniff