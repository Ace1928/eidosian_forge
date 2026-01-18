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
import sys
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
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Type ignores for external libraries without type stubs
import torch
from nltk.corpus import wordnet as wn  # type: ignore
from nltk.corpus.reader.wordnet import Lemma as WNLemma  # type: ignore
from nltk.corpus.reader.wordnet import Synset as WNSynset  # type: ignore
from rdflib import Graph, Literal, URIRef  # type: ignore
from word_forge.utils.nltk_utils import ensure_nltk_data
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
LexicalDataset = Dict[
    str, Union[str, List[Any], WordnetResult, DictResult, ThesaurusResult]
]


# Type definition for data processor functions
class JSONProcessor(Protocol):
    """Protocol defining the contract for JSONL data processing functions."""

    def __call__(self, data: Dict[str, Any]) -> Optional[Union[List[Any], Any]]: ...


# RDF result row type
RDFQueryResult = Tuple[
    Optional[Union[URIRef, Literal]], Optional[Union[URIRef, Literal]]
]
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
    # The type of the result from wn.synsets is not well-defined,
    # but we know it's a list of synset objects
    result = wn.synsets(word, pos, lang, check_exceptions)
    # Filter out None values and only return valid synsets
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


def safely_open_file(
    file_path: PathLike, mode: str = "r", encoding: str = "utf-8"
) -> FileHandle:
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
                            results.extend(cast(Sequence[Any], processed))
                        else:
                            results.append(processed)
                except json.JSONDecodeError:
                    # Skip malformed lines silently
                    pass
    except Exception as e:
        raise LexicalResourceError(
            f"Error processing JSONL file {file_path} (line {line_num}): {str(e)}"
        )

    return results


def read_json_file(file_path: PathLike, default_value: T = None) -> T:
    """
    Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file.
        default_value: Value to return if file doesn't exist or can't be parsed.

    Returns:
        The parsed JSON content or the default value if parsing fails.
    """
    file_handle = safely_open_file(file_path)

    if file_handle is None:
        return default_value

    try:
        with file_handle:
            return cast(T, json.load(file_handle))
    except (json.JSONDecodeError, IOError):
        return default_value


def get_wordnet_data(word: str) -> List[WordnetResult]:
    """
    Extract lexical information using WordNet.

    Args:
        word: The word to look up in WordNet.

    Returns:
        A list of dictionaries containing lexical data for each synset.
    """
    results: List[WordnetResult] = []
    synsets = get_synsets(word)

    for synset in synsets:
        # Cast lemmas to the expected type
        lemmas = cast(List[WNLemma], synset.lemmas())

        # Create lists with proper typing
        synonyms = [cast(str, lemma.name()) for lemma in lemmas]

        # Handle antonyms safely
        antonyms: List[str] = []
        for lemma in lemmas:
            antonym_list = cast(List[WNLemma], lemma.antonyms())
            if antonym_list:
                antonyms.append(cast(str, antonym_list[0].name()))

        # Create the result with explicit casting to satisfy type checker
        data: WordnetResult = {
            "word": word,
            "definition": cast(str, synset.definition()),
            "examples": cast(List[str], synset.examples()),
            "synonyms": synonyms,
            "antonyms": antonyms,
            "part_of_speech": cast(str, synset.pos()),
        }
        results.append(data)
    return results


def get_openthesaurus_data(word: str, openthesaurus_path: PathLike) -> List[str]:
    """
    Extract synonyms from OpenThesaurus data (JSONL format).

    Args:
        word: The word to look up.
        openthesaurus_path: Path to the OpenThesaurus JSONL file.

    Returns:
        A list of synonyms for the word.
    """

    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        if word in data.get("words", []):
            return cast(List[str], data.get("synonyms", []))
        return None

    synonyms: List[str] = []
    for syns in read_jsonl_file(openthesaurus_path, process_line):
        # Explicit casting to satisfy type checker
        synonyms.extend(cast(List[str], syns))

    return list(set(synonyms))


def get_odict_data(word: str, odict_path: PathLike) -> DictResult:
    """
    Extract dictionary definitions from ODict (JSON).

    Args:
        word: The word to look up.
        odict_path: Path to the ODict JSON file.

    Returns:
        Dictionary containing definition and examples.
    """
    default_result: DictResult = {"definition": "Not Found", "examples": []}

    odict_data = read_json_file(odict_path, {})
    return cast(DictResult, odict_data.get(word, default_result))


def get_dbnary_data(word: str, dbnary_path: PathLike) -> List[DbnaryResult]:
    """
    Query Dbnary RDF data using rdflib for definitions and translations.

    Args:
        word: The word to look up.
        dbnary_path: Path to the Dbnary TTL file.

    Returns:
        A list of dictionaries containing definitions and translations.
    """
    if not file_exists(dbnary_path):
        return []

    try:
        graph = Graph()
        graph.parse(str(dbnary_path), format="ttl")

        query = f"""
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?definition ?translation
        WHERE {{
          ?entry ontolex:canonicalForm/ontolex:writtenRep "{word}"@en .
          OPTIONAL {{ ?entry ontolex:definition/rdfs:label ?definition . }}
          OPTIONAL {{ ?entry ontolex:translation/rdfs:label ?translation . }}
        }}
        """
        results = graph.query(query)
        result_list: List[DbnaryResult] = []

        # Explicitly iterate over the result rows and safely extract values
        for row in results:
            # Cast row to the correct type for indexing and handle None values properly
            typed_row = cast(RDFQueryResult, row)
            definition_value = typed_row[0] if typed_row[0] is not None else None
            translation_value = typed_row[1] if typed_row[1] is not None else None

            # Convert to strings, handling specific rdflib types
            definition = str(definition_value) if definition_value is not None else ""
            translation = (
                str(translation_value) if translation_value is not None else ""
            )

            result_list.append(
                {
                    "definition": definition,
                    "translation": translation,
                }
            )
        return result_list
    except Exception as e:
        raise LexicalResourceError(f"Error processing Dbnary data: {str(e)}")


def get_opendictdata(word: str, opendict_path: PathLike) -> DictResult:
    """
    Extract dictionary data from OpenDictData (JSON).

    Args:
        word: The word to look up.
        opendict_path: Path to the OpenDictData JSON file.

    Returns:
        Dictionary containing definition and examples.
    """
    default_result: DictResult = {"definition": "Not Found", "examples": []}

    opendict_data = read_json_file(opendict_path, {})
    return cast(DictResult, opendict_data.get(word, default_result))


def get_thesaurus_data(word: str, thesaurus_path: PathLike) -> List[str]:
    """
    Retrieve synonyms from Thesaurus by Zaibacu (JSONL).

    Args:
        word: The word to look up.
        thesaurus_path: Path to the Thesaurus JSONL file.

    Returns:
        A list of synonyms for the word.
    """

    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        if word == data.get("word"):
            return cast(List[str], data.get("synonyms", []))
        return None

    result_synonyms: List[str] = []
    for syns in read_jsonl_file(thesaurus_path, process_line):
        if isinstance(syns, list):
            result_synonyms.extend(cast(List[str], syns))

    return result_synonyms


class ModelState:
    """Manages transformer model state safely with proper typing."""

    _initialized: bool = False
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    model: Optional[PreTrainedModel] = None

    @classmethod
    def initialize(cls) -> bool:
        """
        Initialize the transformer model and tokenizer.

        Returns:
            True if initialization was successful, False otherwise.
        """
        if cls._initialized:
            return True

        try:
            # Suppress SDPA warning
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            model_name = "qwen/qwen2.5-0.5b-instruct"
            # Direct assignment without explicit casting
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(model_name)
            cls._initialized = True
            return True
        except Exception as e:
            print(f"Warning: Error initializing model: {str(e)}")
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the model is initialized."""
        return cls._initialized


def generate_example_usage(
    word: str,
    definition: str,
    synonyms: List[str],
    antonyms: List[str],
    part_of_speech: str,
) -> str:
    """
    Generate an example sentence using a transformer model.

    Args:
        word: The target word.
        definition: Definition of the word.
        synonyms: List of synonyms.
        antonyms: List of antonyms.
        part_of_speech: Grammatical part of speech.

    Returns:
        Generated example sentence.
    """
    try:
        if not ModelState.initialize():
            return "Model initialization failed."

        if ModelState.tokenizer is None or ModelState.model is None:
            return "Model components unavailable."

        prompt = (
            f"Word: {word}\n"
            f"Part of Speech: {part_of_speech}\n"
            f"Definition: {definition}\n"
            f"Synonyms: {', '.join(synonyms[:5])}\n"
            f"Antonyms: {', '.join(antonyms[:3])}\n"
            f"Task: Generate a single concise example sentence using the word '{word}'.\n"
            f"Example Sentence: "
        )

        # Tokenize the prompt
        tokenizer_output = ModelState.tokenizer(prompt, return_tensors="pt")

        # Generate model output
        outputs = ModelState.model.generate(
            input_ids=tokenizer_output["input_ids"],
            max_new_tokens=50,
            num_beams=3,
            temperature=0.7,
            early_stopping=True,
        )

        # Extract the generated token IDs based on output type
        if isinstance(outputs, torch.Tensor):
            # Direct tensor output (LongTensor)
            sequence = outputs[0]
        else:
            # Structured output (GenerateOutput)
            sequence = outputs.sequences[0]

        # Decode the generated tokens (only once)
        full_text = ModelState.tokenizer.decode(sequence, skip_special_tokens=True)

        # Extract example sentence from the output
        if "Example Sentence:" in full_text:
            parts = full_text.split("Example Sentence:")
            if len(parts) > 1:
                example = parts[1].strip()
                if "." in example:
                    example = example.split(".")[0] + "."
                return example
        return f"Could not generate example for '{word}'."
    except Exception as e:
        return f"Error generating example: {str(e)}"


def truncate_text_at_word_boundary(text: str, max_length: int) -> str:
    """
    Truncate text at a word boundary, ensuring words aren't cut in half.

    Args:
        text: The text to truncate.
        max_length: Maximum allowed length.

    Returns:
        Truncated text ending at a word boundary with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text

    # Find the last space before the max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")

    if last_space > 0:
        return text[:last_space] + "..."

    # If no space found, just truncate at max_length
    return truncated + "..."


def create_lexical_dataset(
    word: str,
    openthesaurus_path: PathLike = "openthesaurus.jsonl",
    odict_path: PathLike = "odict.json",
    dbnary_path: PathLike = "dbnary.ttl",
    opendict_path: PathLike = "opendict.json",
    thesaurus_path: PathLike = "thesaurus.jsonl",
) -> LexicalDataset:
    """
    Create a comprehensive lexical dataset by combining data from all sources.

    Args:
        word: The word to analyze.
        openthesaurus_path: Path to OpenThesaurus data.
        odict_path: Path to ODict data.
        dbnary_path: Path to Dbnary data.
        opendict_path: Path to OpenDictData.
        thesaurus_path: Path to Thesaurus data.

    Returns:
        A dictionary with combined lexical data from all sources.
    """
    # Get WordNet data first as it's the core component
    print("[1/5] Loading WordNet data...")
    wordnet_data = get_wordnet_data(word)
    print(f"Loaded {len(wordnet_data)} WordNet entries")

    # Create the dataset with data from all sources
    data: LexicalDataset = {
        "word": word,
        "wordnet_data": wordnet_data,
        "openthesaurus_synonyms": [],
        "odict_data": {},
        "dbnary_data": {},
        "opendict_data": {},
        "thesaurus_synonyms": [],
    }

    print("[2/5] Loading OpenThesaurus synonyms...")
    data["openthesaurus_synonyms"] = get_openthesaurus_data(word, openthesaurus_path)
    print(f"Loaded {len(data['openthesaurus_synonyms'])} OpenThesaurus synonyms")

    print("[3/5] Loading ODict definitions...")
    data["odict_data"] = get_odict_data(word, odict_path)

    print("[4/5] Loading Dbnary data...")
    data["dbnary_data"] = get_dbnary_data(word, dbnary_path)

    print("[5/5] Loading OpenDictData definitions...")
    data["opendict_data"] = get_opendictdata(word, opendict_path)

    data["thesaurus_synonyms"] = get_thesaurus_data(word, thesaurus_path)

    # Only generate example if we have WordNet data available
    if wordnet_data:
        first_entry = wordnet_data[0]
        example_sentence = generate_example_usage(
            word=word,
            definition=str(first_entry.get("definition", "")),
            synonyms=cast(List[str], data.get("openthesaurus_synonyms", [])),
            antonyms=cast(List[str], first_entry.get("antonyms", [])),
            part_of_speech=str(first_entry.get("part_of_speech", "")),
        )
    else:
        example_sentence = "No example available due to missing WordNet data."

    data["example_sentence"] = example_sentence
    return data


def main() -> None:
    """Main execution entry point."""
    word = sys.argv[1] if len(sys.argv) > 1 else "elucidate"

    print(f"Analyzing lexical data for word: '{word}'")

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    try:
        dataset = create_lexical_dataset(
            word,
            openthesaurus_path=data_dir / "openthesaurus.jsonl",
            odict_path=data_dir / "odict.json",
            dbnary_path=data_dir / "dbnary.ttl",
            opendict_path=data_dir / "opendict.json",
            thesaurus_path=data_dir / "thesaurus.jsonl",
        )

        if "example_sentence" in dataset:
            example = dataset["example_sentence"]
            if isinstance(example, str) and len(example) > 100:
                dataset["example_sentence"] = truncate_text_at_word_boundary(
                    example, 100
                )

        print(json.dumps(dataset, indent=2))

        with open(data_dir / f"{word}_lexical_data.json", "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\nLexical data saved to {data_dir / f'{word}_lexical_data.json'}")
    except LexicalResourceError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
