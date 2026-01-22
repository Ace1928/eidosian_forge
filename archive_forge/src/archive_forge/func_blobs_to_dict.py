import warnings
from collections import OrderedDict
import numpy as np
import xarray as xr
from .. import utils
from .base import dict_to_dataset, generate_dims_coords, make_attrs
from .inference_data import InferenceData
def blobs_to_dict(self):
    """Convert blobs to dictionary {groupname: xr.Dataset}.

        It also stores lp values in sample_stats group.
        """
    store_blobs = self.blob_names is not None
    self.blob_names = [] if self.blob_names is None else self.blob_names
    if self.blob_groups is None:
        self.blob_groups = ['log_likelihood' for _ in self.blob_names]
    if len(self.blob_names) != len(self.blob_groups):
        raise ValueError('blob_names and blob_groups must have the same length, or blob_groups be None')
    if store_blobs:
        if int(self.emcee.__version__[0]) >= 3:
            blobs = self.sampler.get_blobs()
        else:
            blobs = np.array(self.sampler.blobs, dtype=object)
        if (blobs is None or blobs.size == 0) and self.blob_names:
            raise ValueError('No blobs in sampler, blob_names must be None')
        if len(blobs.shape) == 2:
            blobs = np.expand_dims(blobs, axis=-1)
        blobs = blobs.swapaxes(0, 2)
        nblobs, nwalkers, ndraws, *_ = blobs.shape
        if len(self.blob_names) != nblobs and len(self.blob_names) > 1:
            raise ValueError(f'Incorrect number of blob names. Expected {nblobs}, found {len(self.blob_names)}')
    blob_groups_set = set(self.blob_groups)
    blob_groups_set.add('sample_stats')
    idata_groups = ('posterior', 'observed_data', 'constant_data')
    if np.any(np.isin(list(blob_groups_set), idata_groups)):
        raise SyntaxError(f'{idata_groups} groups should not come from blobs. Using them here would overwrite their actual values')
    blob_dict = {group: OrderedDict() for group in blob_groups_set}
    if len(self.blob_names) == 1:
        blob_dict[self.blob_groups[0]][self.blob_names[0]] = blobs.swapaxes(0, 2).swapaxes(0, 1)
    else:
        for i_blob, (name, group) in enumerate(zip(self.blob_names, self.blob_groups)):
            blob = blobs[i_blob]
            if blob.dtype == object:
                blob = blob.reshape(-1)
                blob = np.stack(blob)
                blob = blob.reshape((nwalkers, ndraws, -1))
            blob_dict[group][name] = np.squeeze(blob)
    blob_dict['sample_stats']['lp'] = self.sampler.get_log_prob().swapaxes(0, 1) if hasattr(self.sampler, 'get_log_prob') else self.sampler.lnprobability
    for key, values in blob_dict.items():
        blob_dict[key] = dict_to_dataset(values, library=self.emcee, coords=self.coords, dims=self.dims, index_origin=self.index_origin)
    return blob_dict