from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
@classmethod
def load_local(cls, folder_path: str, *, allow_dangerous_deserialization: bool=False, file_name: str='tfidf_vectorizer') -> TFIDFRetriever:
    """Load the retriever from local storage.

        Args:
            folder_path: Folder path to load from.
            allow_dangerous_deserialization: Whether to allow dangerous deserialization.
                Defaults to False.
                The deserialization relies on .joblib and .pkl files, which can be
                modified to deliver a malicious payload that results in execution of
                arbitrary code on your machine. You will need to set this to `True` to
                use deserialization. If you do this, make sure you trust the source of
                the file.
            file_name: File name to load from. Defaults to "tfidf_vectorizer".

        Returns:
            TFIDFRetriever: Loaded retriever.
        """
    try:
        import joblib
    except ImportError:
        raise ImportError('Could not import joblib, please install with `pip install joblib`.')
    if not allow_dangerous_deserialization:
        raise ValueError('The de-serialization of this retriever is based on .joblib and .pkl files.Such files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine.You will need to set `allow_dangerous_deserialization` to `True` to load this retriever. If you do this, make sure you trust the source of the file, and you are responsible for validating the the file came from a trusted source.')
    path = Path(folder_path)
    vectorizer = joblib.load(path / f'{file_name}.joblib')
    with open(path / f'{file_name}.pkl', 'rb') as f:
        docs, tfidf_array = pickle.load(f)
    return cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array)