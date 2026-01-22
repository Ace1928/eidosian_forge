import re
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import HfApi, HfFolder, get_hf_file_metadata, hf_hub_url
def validate_file_exists(model_name_or_path: Union[str, Path], filename: str, subfolder: str='', revision: Optional[str]=None) -> bool:
    """
    Checks that the file called `filename` exists in the `model_name_or_path` directory or model repo.
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    if model_path.is_dir():
        return (model_path / subfolder / filename).is_file()
    succeeded = True
    try:
        get_hf_file_metadata(hf_hub_url(model_name_or_path, filename, subfolder=subfolder, revision=revision))
    except Exception:
        succeeded = False
    return succeeded