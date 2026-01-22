import logging
from pathlib import Path
from typing import List, Union
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
def maybe_save_preprocessors(src_name_or_path: Union[str, Path], dest_dir: Union[str, Path], src_subfolder: str='', trust_remote_code: bool=False):
    """
    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
        src_subfolder (`str`, defaults to `""`):
            In case the preprocessor files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        trust_remote_code (`bool`, defaults to `False`):
            Whether to allow to save preprocessors that is allowed to run arbitrary code. Use this option at your own risk.
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    for preprocessor in maybe_load_preprocessors(src_name_or_path, subfolder=src_subfolder, trust_remote_code=trust_remote_code):
        preprocessor.save_pretrained(dest_dir)