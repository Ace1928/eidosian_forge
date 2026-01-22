import inspect
import re
import warnings
from functools import wraps
from itertools import chain
from typing import Any, Dict
from ._typing import CallableT
def validate_repo_id(repo_id: str) -> None:
    """Validate `repo_id` is valid.

    This is not meant to replace the proper validation made on the Hub but rather to
    avoid local inconsistencies whenever possible (example: passing `repo_type` in the
    `repo_id` is forbidden).

    Rules:
    - Between 1 and 96 characters.
    - Either "repo_name" or "namespace/repo_name"
    - [a-zA-Z0-9] or "-", "_", "."
    - "--" and ".." are forbidden

    Valid: `"foo"`, `"foo/bar"`, `"123"`, `"Foo-BAR_foo.bar123"`

    Not valid: `"datasets/foo/bar"`, `".repo_id"`, `"foo--bar"`, `"foo.git"`

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_repo_id
    >>> validate_repo_id(repo_id="valid_repo_id")
    >>> validate_repo_id(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
    ```

    Discussed in https://github.com/huggingface/huggingface_hub/issues/1008.
    In moon-landing (internal repository):
    - https://github.com/huggingface/moon-landing/blob/main/server/lib/Names.ts#L27
    - https://github.com/huggingface/moon-landing/blob/main/server/views/components/NewRepoForm/NewRepoForm.svelte#L138
    """
    if not isinstance(repo_id, str):
        raise HFValidationError(f"Repo id must be a string, not {type(repo_id)}: '{repo_id}'.")
    if repo_id.count('/') > 1:
        raise HFValidationError(f"Repo id must be in the form 'repo_name' or 'namespace/repo_name': '{repo_id}'. Use `repo_type` argument if needed.")
    if not REPO_ID_REGEX.match(repo_id):
        raise HFValidationError(f"Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: '{repo_id}'.")
    if '--' in repo_id or '..' in repo_id:
        raise HFValidationError(f"Cannot have -- or .. in repo_id: '{repo_id}'.")
    if repo_id.endswith('.git'):
        raise HFValidationError(f"Repo_id cannot end by '.git': '{repo_id}'.")