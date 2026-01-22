from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def copy_scan(src_scan, dst_scan, scan_cache_dir):
    """Copy scan from source XNAT to destination XNAT"""
    scan_type = src_scan.datatype()
    if scan_type == '':
        scan_type = 'xnat:otherDicomScanData'
    dst_scan.create(scans=scan_type)
    copy_attributes(src_scan, dst_scan)
    for src_res in src_scan.resources().fetchall('obj'):
        res_label = src_res.label()
        print('INFO:Processing resource:%s...' % res_label)
        dst_res = dst_scan.resource(res_label)
        res_cache_dir = op.join(scan_cache_dir, res_label)
        if res_label == 'SNAPSHOTS':
            copy_res(src_res, dst_res, res_cache_dir)
        else:
            copy_res(src_res, dst_res, res_cache_dir, use_zip=True)