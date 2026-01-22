import argparse
import os
import sys
import nibabel as nb
def lossless_slice(img, slicers):
    if not nb.imageclasses.spatial_axes_first(img):
        raise ValueError('Cannot slice an image that is not known to have spatial axes first')
    scaling = hasattr(img.header, 'set_slope_inter')
    data = img.dataobj._get_unscaled(slicers) if scaling else img.dataobj[slicers]
    roi_img = img.__class__(data, affine=img.slicer.slice_affine(slicers), header=img.header)
    if scaling:
        roi_img.header.set_slope_inter(img.dataobj.slope, img.dataobj.inter)
    return roi_img