import csv
import os
import sys
from optparse import Option, OptionParser
import numpy as np
import numpy.linalg as npl
import nibabel
import nibabel.nifti1 as nifti1
import nibabel.parrec as pr
from nibabel.affines import apply_affine, from_matvec, to_matvec
from nibabel.filename_parser import splitext_addext
from nibabel.mriutils import MRIError, calculate_dwell_time
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation
from nibabel.parrec import one_line
from nibabel.volumeutils import fname_ext_ul_case
def proc_file(infile, opts):
    basefilename = splitext_addext(os.path.basename(infile))[0]
    if opts.outdir is not None:
        basefilename = os.path.join(opts.outdir, basefilename)
    if opts.compressed:
        verbose('Using gzip compression')
        outfilename = basefilename + '.nii.gz'
    else:
        outfilename = basefilename + '.nii'
    if os.path.isfile(outfilename) and (not opts.overwrite):
        raise OSError(f'Output file "{outfilename}" exists, use --overwrite to overwrite it')
    scaling = 'dv' if opts.scaling == 'off' else opts.scaling
    infile = fname_ext_ul_case(infile)
    pr_img = pr.load(infile, permit_truncated=opts.permit_truncated, scaling=scaling, strict_sort=opts.strict_sort)
    pr_hdr = pr_img.header
    affine = pr_hdr.get_affine(origin=opts.origin)
    slope, intercept = pr_hdr.get_data_scaling(scaling)
    if opts.scaling != 'off':
        verbose(f'Using data scaling "{opts.scaling}"')
    if opts.scaling == 'off':
        slope = np.array([1.0])
        intercept = np.array([0.0])
        in_data = pr_img.dataobj.get_unscaled()
        out_dtype = pr_hdr.get_data_dtype()
    elif not np.any(np.diff(slope)) and (not np.any(np.diff(intercept))):
        slope = slope.ravel()[0]
        intercept = intercept.ravel()[0]
        in_data = pr_img.dataobj.get_unscaled()
        out_dtype = pr_hdr.get_data_dtype()
    else:
        slope = np.array([1.0])
        intercept = np.array([0.0])
        in_data = np.array(pr_img.dataobj)
        out_dtype = np.float64
    ornt = io_orientation(np.diag([-1, 1, 1, 1]).dot(affine))
    if np.array_equal(ornt, [[0, 1], [1, 1], [2, 1]]):
        t_aff = np.eye(4)
    else:
        t_aff = inv_ornt_aff(ornt, pr_img.shape)
        affine = np.dot(affine, t_aff)
        in_data = apply_orientation(in_data, ornt)
    bvals, bvecs = pr_hdr.get_bvals_bvecs()
    if not opts.keep_trace:
        if bvecs is not None:
            bad_mask = np.logical_and(bvals != 0, (bvecs == 0).all(axis=1))
            if bad_mask.sum() > 0:
                pl = 's' if bad_mask.sum() != 1 else ''
                verbose(f'Removing {bad_mask.sum()} DTI trace volume{pl}')
                good_mask = ~bad_mask
                in_data = in_data[..., good_mask]
                bvals = bvals[good_mask]
                bvecs = bvecs[good_mask]
    nimg = nifti1.Nifti1Image(in_data, affine, pr_hdr)
    nhdr = nimg.header
    nhdr.set_data_dtype(out_dtype)
    nhdr.set_slope_inter(slope, intercept)
    nhdr.set_sform(affine, code=1)
    nhdr.set_qform(affine, code=1)
    if 'parse' in opts.minmax:
        verbose('Loading (and scaling) the data to determine value range')
    if opts.minmax[0] == 'parse':
        nhdr['cal_min'] = in_data.min() * slope + intercept
    else:
        nhdr['cal_min'] = float(opts.minmax[0])
    if opts.minmax[1] == 'parse':
        nhdr['cal_max'] = in_data.max() * slope + intercept
    else:
        nhdr['cal_max'] = float(opts.minmax[1])
    if opts.store_header:
        with open(infile, 'rb') as fobj:
            hdr_dump = fobj.read()
            dump_ext = nifti1.Nifti1Extension('comment', hdr_dump)
        nhdr.extensions.append(dump_ext)
    verbose(f'Writing {outfilename}')
    nibabel.save(nimg, outfilename)
    if opts.bvs:
        if bvals is None and bvecs is None:
            verbose('No DTI volumes detected, bvals and bvecs not written')
        elif bvecs is None:
            verbose('DTI volumes detected, but no diffusion direction info wasfound.  Writing .bvals file only.')
            with open(basefilename + '.bvals', 'w') as fid:
                for val in bvals:
                    fid.write(f'{val} ')
                fid.write('\n')
        else:
            verbose('Writing .bvals and .bvecs files')
            orig2new = npl.inv(t_aff)
            bv_reorient = from_matvec(to_matvec(orig2new)[0], [0, 0, 0])
            bvecs = apply_affine(bv_reorient, bvecs)
            with open(basefilename + '.bvals', 'w') as fid:
                for val in bvals:
                    fid.write(f'{val} ')
                fid.write('\n')
            with open(basefilename + '.bvecs', 'w') as fid:
                for row in bvecs.T:
                    for val in row:
                        fid.write(f'{val} ')
                    fid.write('\n')
    if opts.vol_info:
        labels = pr_img.header.get_volume_labels()
        if len(labels) > 0:
            vol_keys = list(labels.keys())
            with open(basefilename + '.ordering.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(vol_keys)
                for vals in zip(*[labels[k] for k in vol_keys]):
                    csvwriter.writerow(vals)
    if opts.dwell_time:
        try:
            dwell_time = calculate_dwell_time(pr_hdr.get_water_fat_shift(), pr_hdr.get_echo_train_length(), opts.field_strength)
        except MRIError:
            verbose('No EPI factors, dwell time not written')
        else:
            verbose(f'Writing dwell time ({dwell_time!r} sec) calculated assuming {opts.field_strength}T magnet')
            with open(basefilename + '.dwell_time', 'w') as fid:
                fid.write(f'{dwell_time!r}\n')