import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def read_mosaic_dir(dicom_path, globber='*.dcm', check_is_dwi=False, dicom_kwargs=None):
    """Read all Siemens mosaic DICOMs in directory, return arrays, params

    Parameters
    ----------
    dicom_path : str
       path containing mosaic DICOM images
    globber : str, optional
       glob to apply within `dicom_path` to select DICOM files.  Default
       is ``*.dcm``
    check_is_dwi : bool, optional
       If True, raises an error if we don't find DWI information in the
       DICOM headers.
    dicom_kwargs : None or dict
       Extra keyword arguments to pass to the pydicom ``dcmread`` function.

    Returns
    -------
    data : 4D array
       data array with last dimension being acquisition. If there were N
       acquisitions, each of shape (X, Y, Z), `data` will be shape (X,
       Y, Z, N)
    affine : (4,4) array
       affine relating 3D voxel space in data to RAS world space
    b_values : (N,) array
       b values for each acquisition.  nan if we did not find diffusion
       information for these images.
    unit_gradients : (N, 3) array
       gradient directions of unit length for each acquisition.  (nan,
       nan, nan) if we did not find diffusion information.
    """
    if dicom_kwargs is None:
        dicom_kwargs = {}
    full_globber = pjoin(dicom_path, globber)
    filenames = sorted(glob.glob(full_globber))
    b_values = []
    gradients = []
    arrays = []
    if len(filenames) == 0:
        raise OSError(f'Found no files with "{full_globber}"')
    for fname in filenames:
        dcm_w = wrapper_from_file(fname, **dicom_kwargs)
        if not dcm_w.is_mosaic:
            raise DicomReadError('data does not appear to be in mosaic format')
        arrays.append(dcm_w.get_data()[..., None])
        q = dcm_w.q_vector
        if q is None:
            if check_is_dwi:
                raise DicomReadError(f'Could not find diffusion information reading file "{fname}";  is it possible this is not a _raw_ diffusion directory? Could it be a processed dataset like ADC etc?')
            b = np.nan
            g = np.ones((3,)) + np.nan
        else:
            b = dcm_w.b_value
            g = dcm_w.b_vector
        b_values.append(b)
        gradients.append(g)
    affine = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return (np.concatenate(arrays, -1), affine, np.array(b_values), np.array(gradients))