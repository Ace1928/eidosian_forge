#!/usr/bin/env python3
"""
Comprehensive Lexical and Linguistic Dataset Creation Script

This script integrates multiple open-source lexical resources including WordNet,
OpenThesaurus, ODict, Dbnary, OpenDictData, and Thesaurus by Zaibacu to create a
comprehensive lexical dataset for a given word. It also generates example sentences
using the transformer model (qwen2.5-0.5b-instruct).

EIDOSIAN CODE POLISHING PROTOCOL v3.14.15 applied.
"""

import json
import os
from os import PathLike as OSPathLike
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Heavy dependencies are optional and loaded lazily to avoid import-time failures
try:
    import torch
except Exception:  # pragma: no cover - optional heavy dependency
    torch = None  # type: ignore
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    from nltk.corpus.reader.wordnet import Lemma as WNLemma  # type: ignore
    from nltk.corpus.reader.wordnet import Synset as WNSynset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wn = None  # type: ignore
    WNLemma = None  # type: ignore
    WNSynset = None  # type: ignore
try:
    from rdflib import Graph, Literal, URIRef  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Graph = Literal = URIRef = None  # type: ignore
from word_forge.utils.nltk_utils import ensure_nltk_data

try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    from transformers.generation.utils import (  # type: ignore
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    PreTrainedModel = PreTrainedTokenizer = PreTrainedTokenizerFast = None  # type: ignore
    GenerateBeamDecoderOnlyOutput = GenerateBeamEncoderDecoderOutput = None  # type: ignore
    GenerateDecoderOnlyOutput = GenerateEncoderDecoderOutput = None  # type: ignore

if TYPE_CHECKING:
    # These imports are only for type checking, not at runtime
    pass

# Type definitions
PathLike = Union[str, Path, OSPathLike[str]]
T = TypeVar("T")  # Generic type
FileHandle = Optional[Union[TextIO, IO[Any]]]


# Enhanced type definitions for domain-specific structures
class Lemma(Protocol):
    """Protocol defining the interface for WordNet lemmas."""

    def name(self) -> str: ...
    def antonyms(self) -> List["Lemma"]: ...


# Define custom result types
WordnetResult = Dict[str, Union[str, List[str]]]
DbnaryResult = Dict[str, str]
ThesaurusResult = List[str]
DictResult = Dict[str, Union[str, List[str]]]
LexicalDataset = Dict[str, Union[str, List[Any], WordnetResult, DictResult, ThesaurusResult]]


# Type definition for data processor functions
class JSONProcessor(Protocol):
    """Protocol defining the contract for JSONL data processing functions."""

    def __call__(self, data: Dict[str, Any]) -> Optional[Union[List[Any], Any]]: ...


# RDF result row type
RDFQueryResult = Tuple[Optional[Union[URIRef, Literal]], Optional[Union[URIRef, Literal]]]
GenerateOutput = Union[
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
]


# Custom exceptions for better error handling
class LexicalResourceError(Exception):
    """Exception raised when a lexical resource cannot be accessed or processed."""

    pass


def get_synsets(
    word: str,
    pos: Optional[str] = None,
    lang: str = "eng",
    check_exceptions: bool = True,
) -> List[WNSynset]:
    """
    Typed wrapper for wordnet.synsets to satisfy type checking.

    Args:
        word: The word to look up in WordNet.
        pos: Part of speech filter (n, v, a, r).
        lang: Language code.
        check_exceptions: Whether to check for exceptions.

    Returns:
        List of synsets for the word.
    """
    ensure_nltk_data()
    if wn is None:
        return []
    result = wn.synsets(word, pos, lang, check_exceptions)
    return [synset for synset in result if synset is not None]


def file_exists(file_path: PathLike) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to check.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(str(file_path))


def safely_open_file(file_path: PathLike, mode: str = "r", encoding: str = "utf-8") -> FileHandle:
    """
    Safely open a file with proper error handling.

    Args:
        file_path: Path to the file to open.
        mode: File open mode ('r', 'w', etc.).
        encoding: Character encoding to use.

    Returns:
        File handle if file exists and can be opened, None if file doesn't exist.

    Raises:
        LexicalResourceError: If file exists but cannot be opened due to permissions or other issues.
    """
    if not file_exists(file_path):
        return None

    try:
        return cast(FileHandle, open(str(file_path), mode, encoding=encoding))
    except (IOError, OSError) as e:
        raise LexicalResourceError(f"Error opening file {file_path}: {str(e)}")


def read_jsonl_file(file_path: PathLike, process_func: JSONProcessor) -> List[Any]:
    """
    Read a JSONL file and process each line with the provided function.

    Args:
        file_path: Path to the JSONL file.
        process_func: Function to process each JSON object from a line.

    Returns:
        List of all processed results from valid lines.

    Raises:
        LexicalResourceError: If file reading fails after opening.
    """
    results: List[Any] = []
    file_handle = safely_open_file(file_path)

    if file_handle is None:
        return results

    line_num = 0
    try:
        with file_handle:
            for line_num, line in enumerate(file_handle, 1):
                try:
                    data = json.loads(line)
                    processed = process_func(data)
                    if processed is not None:
                        if isinstance(processed, list):
                            results.extend(processed)
                        else:
                            results.append(processed)
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError) as e:
        raise LexicalResourceError(f"Error reading JSONL file {file_path} at line {line_num}: {str(e)}")

    return results
