import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def make_fcpl(track_order=False, fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None):
    """ Set up a file creation property list """
    if track_order or fs_strategy:
        plist = h5p.create(h5p.FILE_CREATE)
        if track_order:
            plist.set_link_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
            plist.set_attr_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
        if fs_strategy:
            strategies = {'fsm': h5f.FSPACE_STRATEGY_FSM_AGGR, 'page': h5f.FSPACE_STRATEGY_PAGE, 'aggregate': h5f.FSPACE_STRATEGY_AGGR, 'none': h5f.FSPACE_STRATEGY_NONE}
            fs_strat_num = strategies.get(fs_strategy, -1)
            if fs_strat_num == -1:
                raise ValueError('Invalid file space strategy type')
            plist.set_file_space_strategy(fs_strat_num, fs_persist, fs_threshold)
            if fs_page_size and fs_strategy == 'page':
                plist.set_file_space_page_size(int(fs_page_size))
    else:
        plist = None
    return plist