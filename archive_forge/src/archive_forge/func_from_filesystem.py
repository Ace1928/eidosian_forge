from __future__ import annotations
import concurrent.futures
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import (
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.registry import get_parser
@classmethod
def from_filesystem(cls, path: _PathLike, *, glob: str='**/[!.]*', exclude: Sequence[str]=(), suffixes: Optional[Sequence[str]]=None, show_progress: bool=False, parser: Union[DEFAULT, BaseBlobParser]='default', num_workers: int=4, parser_kwargs: Optional[dict]=None) -> ConcurrentLoader:
    """Create a concurrent generic document loader using a filesystem blob loader.

        Args:
            path: The path to the directory to load documents from.
            glob: The glob pattern to use to find documents.
            suffixes: The suffixes to use to filter documents. If None, all files
                      matching the glob will be loaded.
            exclude: A list of patterns to exclude from the loader.
            show_progress: Whether to show a progress bar or not (requires tqdm).
                           Proxies to the file system loader.
            parser: A blob parser which knows how to parse blobs into documents
            num_workers: Max number of concurrent workers to use.
            parser_kwargs: Keyword arguments to pass to the parser.
        """
    blob_loader = FileSystemBlobLoader(path, glob=glob, exclude=exclude, suffixes=suffixes, show_progress=show_progress)
    if isinstance(parser, str):
        if parser == 'default' and cls.get_parser != GenericLoader.get_parser:
            blob_parser = cls.get_parser(**parser_kwargs or {})
        else:
            blob_parser = get_parser(parser)
    else:
        blob_parser = parser
    return cls(blob_loader, blob_parser, num_workers=num_workers)