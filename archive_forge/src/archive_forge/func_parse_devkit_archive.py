import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def parse_devkit_archive(root: str, file: Optional[str]=None) -> None:
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
        metafile = os.path.join(devkit_root, 'data', 'meta.mat')
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return (idx_to_wnid, wnid_to_classes)

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, 'data', 'ILSVRC2012_validation_ground_truth.txt')
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)
    archive_meta = ARCHIVE_META['devkit']
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    _verify_archive(root, file, md5)
    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)
        devkit_root = os.path.join(tmp_dir, 'ILSVRC2012_devkit_t12')
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))