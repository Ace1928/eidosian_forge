from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def ring_contains_point(coords, p):
    """Fast point-in-polygon crossings algorithm, MacMartin optimization.

    Adapted from code by Eric Haynes
    http://www.realtimerendering.com/resources/GraphicsGems//gemsiv/ptpoly_haines/ptinpoly.c
    
    Original description:
        Shoot a test ray along +X axis.  The strategy, from MacMartin, is to
        compare vertex Y values to the testing point's Y and quickly discard
        edges which are entirely to one side of the test ray.
    """
    tx, ty = p
    vtx0 = coords[0]
    yflag0 = vtx0[1] >= ty
    inside_flag = False
    for vtx1 in coords[1:]:
        yflag1 = vtx1[1] >= ty
        if yflag0 != yflag1:
            xflag0 = vtx0[0] >= tx
            if xflag0 == (vtx1[0] >= tx):
                if xflag0:
                    inside_flag = not inside_flag
            elif vtx1[0] - (vtx1[1] - ty) * (vtx0[0] - vtx1[0]) / (vtx0[1] - vtx1[1]) >= tx:
                inside_flag = not inside_flag
        yflag0 = yflag1
        vtx0 = vtx1
    return inside_flag