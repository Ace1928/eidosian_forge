# filepath: /home/lloyd/eidosian_forge/word_forge/src/word_forge/parser/lexical_functions.py
# ============================================================================
#                              IMPORTS
# ============================================================================
import functools
import json
import os
import re
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Union

from eidosian_core import eidosian
from nltk.corpus import wordnet as wn  # type: ignore
from nltk.corpus.reader.wordnet import Lemma, Synset  # type: ignore

# Optional dependency for DBnary RDF processing
try:
    from rdflib import Graph
    from rdflib import Literal as RdfLiteral
    from rdflib.query import ResultRow

    _rdflib_available = True
except ImportError:
    _rdflib_available = False
    Graph = None  # type: ignore[assignment]
    RdfLiteral = None  # type: ignore[assignment]
    ResultRow = None  # type: ignore[assignment]

from word_forge.configs.config_essentials import (
    DbnaryEntry,
    DictionaryEntry,
    JsonData,
    LexicalDataset,
    LexicalResourceError,
    ResourceParsingError,
    T,
    WordnetEntry,
)
from word_forge.parser.language_model import ModelState
from word_forge.parser.structured_validator import validated_query
from word_forge.utils.nltk_utils import ensure_nltk_data


# ============================================================================
#                           FILE OPERATIONS
# ============================================================================
@eidosian()
def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists at the specified path.

    Args:
        file_path: Path to check for file existence

    Returns:
        True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)


@eidosian()
@contextmanager
def safe_open(file_path: Union[str, Path], mode: str = "r", encoding: str = "utf-8") -> Iterator[Optional[IO[Any]]]:
    """
    Safely open a file, handling non-existent files and IO errors.

    Args:
        file_path: Path to the file to open
        mode: File mode (r, w, etc.)
        encoding: Text encoding to use

    Yields:
        File handle if file exists and can be opened, None otherwise

    Raises:
        LexicalResourceError: If file exists but cannot be opened due to IO errors
    """
    if not file_exists(file_path):
        yield None
        return

    try:
        with open(file_path, mode, encoding=encoding) as f:
            fh: IO[Any] = f
            yield fh
    except (IOError, OSError) as e:
        raise LexicalResourceError(f"Error opening file {file_path}: {str(e)}")


@eidosian()
def read_json_file(file_path: Union[str, Path], default_value: T = None) -> Union[JsonData, T]:
    """
    Read and parse a JSON file, returning a default value if the file doesn't exist or is invalid.

    Args:
        file_path: Path to the JSON file
        default_value: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or the default value

    Raises:
        LexicalResourceError: If file exists but cannot be opened
    """
    with safe_open(file_path) as fh:
        if fh is None:
            return default_value
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return default_value


@eidosian()
def read_jsonl_file(file_path: Union[str, Path], process_func: Callable[[Dict[str, Any]], Optional[T]]) -> List[T]:
    """
    Read and process a JSON Lines file line by line.

    Args:
        file_path: Path to the JSONL file
        process_func: Function to process each parsed JSON line

    Returns:
        List of processed results

    Raises:
        LexicalResourceError: If file cannot be accessed or processing fails
    """
    results: List[T] = []
    with safe_open(file_path) as fh:
        if fh is None:
            return results

        line_num = 0
        try:
            for line in fh:
                line_num += 1
                if not line.strip():
                    continue

                data = json.loads(line)
                processed = process_func(data)
                if processed is not None:
                    results.append(processed)
        except Exception as e:
            raise ResourceParsingError(f"Error processing line {line_num} in {file_path}: {str(e)}")

    return results


# ============================================================================
#                            WORDNET FUNCTIONS
# ============================================================================
@eidosian()
@functools.lru_cache(maxsize=1024)
def get_synsets(word: str) -> List[Synset]:
    """Retrieve synsets from WordNet for a given word with efficient caching."""
    ensure_nltk_data()
    return wn.synsets(word)  # type: ignore


@eidosian()
def get_wordnet_data(word: str) -> List[WordnetEntry]:
    """
    Extract comprehensive linguistic data from WordNet for a given word.

    Args:
        word: Word to retrieve data for

    Returns:
        List of structured entries containing definitions, examples, synonyms, antonyms,
        and part-of-speech information
    """
    results: List[WordnetEntry] = []
    synsets: List[Synset] = get_synsets(word)

    for synset in synsets:
        lemmas: List[Lemma] = synset.lemmas() or []
        synonyms: List[str] = []
        for lemma in lemmas:
            name = lemma.name()
            if isinstance(name, str):
                synonyms.append(name.replace("_", " "))

        # Extract antonyms from lemmas
        antonyms: List[str] = []
        for lemma in lemmas:
            lemma_antonyms: List[Lemma] = lemma.antonyms()
            for antonym in lemma_antonyms:
                antonym_name = antonym.name()
                if isinstance(antonym_name, str):
                    antonym_name = antonym_name.replace("_", " ")
                    antonyms.append(antonym_name)

        # Explicitly type with expected return types but handle variations
        # Cast the result to Optional[str] as nltk types might be incomplete
        definition_result: Optional[str] = typing.cast(Optional[str], synset.definition())
        definition: str = definition_result if definition_result is not None else ""

        # Cast the result to Optional[List[str]]
        examples_result: Optional[List[str]] = typing.cast(Optional[List[str]], synset.examples())
        examples: List[str] = examples_result if examples_result is not None else []

        # Cast the result to Optional[str]
        pos_result: Optional[str] = typing.cast(Optional[str], synset.pos())
        pos: str = pos_result if pos_result is not None else ""

        results.append(
            WordnetEntry(
                word=word,
                definition=definition,
                examples=examples,
                synonyms=synonyms,
                antonyms=antonyms,
                part_of_speech=pos,
            )
        )

    return results


# ============================================================================
#                          LEXICAL DATA SOURCES
# ============================================================================
@eidosian()
def get_openthesaurus_data(word: str, openthesaurus_path: str) -> List[str]:
    """
    Extract synonyms from OpenThesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        openthesaurus_path: Path to the OpenThesaurus JSONL file

    Returns:
        List of unique synonyms with duplicates removed while preserving order
    """

    @eidosian()
    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        words = data.get("words", [])
        if word in words:
            return [w for w in words if w != word]
        return None

    synonyms: List[str] = []
    for syns in read_jsonl_file(openthesaurus_path, process_line):
        synonyms.extend(syns)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(synonyms))


@eidosian()
def get_odict_data(word: str, odict_path: str) -> DictionaryEntry:
    """
    Retrieve dictionary data from ODict for a given word.

    Args:
        word: Word to retrieve data for
        odict_path: Path to the ODict JSON file

    Returns:
        Dictionary containing definition and usage examples
    """
    default_entry: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    odict_data = read_json_file(odict_path, {})
    entry = odict_data.get(word, default_entry) if isinstance(odict_data, dict) else default_entry
    if isinstance(entry, dict) and "definition" in entry and "examples" in entry:
        examples_raw = entry.get("examples", [])
        if isinstance(examples_raw, list):
            examples = [str(ex) for ex in examples_raw]
        else:
            examples = []
        return DictionaryEntry(definition=str(entry["definition"]), examples=examples)
    return default_entry


@eidosian()
def get_dbnary_data(word: str, dbnary_path: str) -> List[DbnaryEntry]:
    """
    Extract linguistic data from DBnary RDF for a given word.

    Args:
        word: Word to retrieve data for
        dbnary_path: Path to the DBnary TTL file

    Returns:
        List of entries containing definitions and translations

    Raises:
        LexicalResourceError: If there's an error processing the DBnary data
    """
    if not _rdflib_available:
        return []

    if not file_exists(dbnary_path):
        return []

    try:
        graph = Graph()
        graph.parse(dbnary_path, format="ttl")

        sparql_query = f"""
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?definition ?translation
        WHERE {{
          ?entry ontolex:canonicalForm/ontolex:writtenRep "{word}"@en .
          OPTIONAL {{ ?entry ontolex:definition/rdfs:label ?definition . }}
          OPTIONAL {{ ?entry ontolex:translation/rdfs:label ?translation . }}
        }}
        """

        results = graph.query(sparql_query)
        output: List[DbnaryEntry] = []

        for row in results:
            # Ensure row is a ResultRow before accessing elements by index/key
            if not isinstance(row, ResultRow):
                continue

            # Access elements safely using .get() or check length
            definition_node = row.definition if hasattr(row, "definition") else None
            translation_node = row.translation if hasattr(row, "translation") else None

            # Convert rdflib Nodes (Literal, URIRef) to string safely
            definition = str(definition_node.value) if isinstance(definition_node, RdfLiteral) else ""
            translation = str(translation_node.value) if isinstance(translation_node, RdfLiteral) else ""

            if definition or translation:
                output.append(DbnaryEntry(definition=definition, translation=translation))

        return output

    except Exception as e:
        raise LexicalResourceError(f"Error processing Dbnary data: {str(e)}")


@eidosian()
def get_opendictdata(word: str, opendict_path: str) -> DictionaryEntry:
    """
    Retrieve dictionary data from OpenDict for a given word.

    Args:
        word: Word to retrieve data for
        opendict_path: Path to the OpenDict JSON file

    Returns:
        Dictionary containing definition and examples
    """
    default_entry: DictionaryEntry = {
        "definition": "Not Found",
        "examples": [],
    }
    data = read_json_file(opendict_path, {})
    entry = data.get(word, default_entry) if isinstance(data, dict) else default_entry
    if isinstance(entry, dict) and "definition" in entry and "examples" in entry:
        # Ensure examples are processed correctly into List[str]
        examples_raw = entry.get("examples", [])
        examples: List[str] = []
        if isinstance(examples_raw, list):
            examples = [str(ex) for ex in examples_raw if isinstance(ex, (str, int, float))]
        return DictionaryEntry(definition=str(entry["definition"]), examples=examples)
    return default_entry


@eidosian()
def get_thesaurus_data(word: str, thesaurus_path: str) -> List[str]:
    """
    Extract synonyms from Thesaurus for a given word.

    Args:
        word: Word to retrieve synonyms for
        thesaurus_path: Path to the Thesaurus JSONL file

    Returns:
        List of synonyms from the thesaurus source
    """

    @eidosian()
    def process_line(data: Dict[str, Any]) -> Optional[List[str]]:
        if word == data.get("word"):
            return data.get("synonyms", [])
        return None

    results: List[str] = []
    for syns in read_jsonl_file(thesaurus_path, process_line):
        results.extend(syns)

    return results


# ============================================================================
#                       EXAMPLE GENERATION FUNCTIONS
# ============================================================================
@eidosian()
def generate_example_usage(
    word: str,
    definition: str,
    synonyms: List[str],
    antonyms: List[str],
    pos: str,
    model_state: ModelState,
) -> str:
    """
    Generate an example sentence for a word using a language model.

    Args:
        word: The target word to use in the example
        definition: The word's definition
        synonyms: List of word synonyms
        antonyms: List of word antonyms
        pos: Part of speech
        model_state: Initialized :class:`ModelState` instance to use for text generation

    Returns:
        A generated example sentence or an error message
    """
    # Construct prompt with word details
    prompt = (
        f"Word: {word}\n"
        f"Part of Speech: {pos}\n"
        f"Definition: {definition}\n"
        f"Synonyms: {', '.join(synonyms[:5])}\n"
        f"Antonyms: {', '.join(antonyms[:3])}\n"
        f"Task: Generate a single concise example sentence using the word '{word}'.\n"
        f"Example Sentence: "
    )

    # Use the provided model state for generation
    full_text = model_state.generate_text(prompt)

    if not full_text:
        return f"Could not generate example for '{word}'."

    # Parse out just the generated example
    if "Example Sentence:" in full_text:
        parts = full_text.split("Example Sentence:")
        if len(parts) > 1:
            example = parts[1].strip()
            # Capture up to first period for a complete sentence
            if "." in example:
                sentence_end = example.find(".") + 1
                return example[:sentence_end].strip()
            return example

    # If we got text but couldn't parse it properly, return it as-is
    if full_text and not full_text.startswith("Could not"):
        # Try to find the first complete sentence
        sentences = re.split(r"[.!?]", full_text)
        if sentences and len(sentences[0]) > 5:  # Minimum length for a valid sentence
            return sentences[0].strip() + "."
        return full_text.strip()

    return f"Could not extract valid example for '{word}'."


@eidosian()
def generate_comprehensive_enrichment(
    word: str,
    definition: str,
    pos: str,
    model_state: ModelState,
) -> Dict[str, Any]:
    """
    Generate comprehensive lexical and semantic enrichment using the LLM.

    Extracts holonyms, meronyms, hypernyms, hyponyms, and emotional dimensions.
    Utilizes validated_query for schema enforcement and retries.

    Args:
        word: Target word
        definition: Known definition
        pos: Part of speech
        model_state: Language model state

    Returns:
        Dictionary containing enriched data fields.
    """
    prompt = (
        f"Analyze the word '{word}' ({pos}: {definition}).\n"
        "Provide a JSON object with the following schema:\n"
        "{\n"
        '  "word": "string",\n'
        '  "definition": "string",\n'
        '  "synonyms": ["list of strings"],\n'
        '  "antonyms": ["list of strings"],\n'
        '  "hypernyms": ["list of strings"],\n'
        '  "hyponyms": ["list of strings"],\n'
        '  "holonyms": ["list of strings"],\n'
        '  "meronyms": ["list of strings"],\n'
        '  "emotional_valence": float (-1.0 to 1.0),\n'
        '  "emotional_arousal": float (0.0 to 1.0),\n'
        '  "connotation": "positive|negative|neutral",\n'
        '  "usage_examples": ["list of strings"]\n'
        "}\n"
        "Return ONLY valid JSON."
    )

    result = validated_query(model_state=model_state, prompt=prompt, context_word=word, max_retries=2)

    if result.is_success:
        return result.unwrap()
    else:
        # Fallback to empty if even validated_query fails
        return {}


# ============================================================================
#                          DATASET CREATION
# ============================================================================
@eidosian()
def create_lexical_dataset(
    word: str,
    openthesaurus_path: str = "data/openthesaurus.jsonl",
    odict_path: str = "data/odict.json",
    dbnary_path: str = "data/dbnary.ttl",
    opendict_path: str = "data/opendict.json",
    thesaurus_path: str = "data/thesaurus.jsonl",
    model_state: ModelState = None,
) -> LexicalDataset:
    """
    Create a comprehensive dataset of lexical information for a word.

    Args:
        word: The word to gather data for
        openthesaurus_path: Path to OpenThesaurus data
        odict_path: Path to ODict data
        dbnary_path: Path to DBnary data
        opendict_path: Path to OpenDict data
        thesaurus_path: Path to Thesaurus data
        model_state: Optional :class:`ModelState` used to generate example sentences

    Returns:
        Dictionary containing comprehensive lexical data from all sources
    """
    llm_enabled = model_state is not None

    wordnet_data = get_wordnet_data(word)

    dataset: LexicalDataset = {
        "word": word,
        "wordnet_data": wordnet_data,
        "openthesaurus_synonyms": get_openthesaurus_data(word, openthesaurus_path),
        "odict_data": get_odict_data(word, odict_path),
        "dbnary_data": get_dbnary_data(word, dbnary_path),
        "opendict_data": get_opendictdata(word, opendict_path),
        "thesaurus_synonyms": get_thesaurus_data(word, thesaurus_path),
        "example_sentence": "",
    }

    # Generate an example sentence if WordNet data exists
    if wordnet_data and llm_enabled:
        first_entry = wordnet_data[0]
        example = generate_example_usage(
            word,
            definition=first_entry.get("definition", ""),
            synonyms=dataset["openthesaurus_synonyms"],
            antonyms=first_entry.get("antonyms", []),
            pos=first_entry.get("part_of_speech", ""),
            model_state=model_state,
        )
        dataset["example_sentence"] = example
    else:
        dataset["example_sentence"] = ""

    return dataset


# ============================================================================
#                                 EXPORTS
# ============================================================================
__all__ = [
    # File operations
    "file_exists",
    "safe_open",
    "read_json_file",
    "read_jsonl_file",
    # WordNet functions
    "get_synsets",
    "get_wordnet_data",
    # Lexical data sources
    "get_openthesaurus_data",
    "get_odict_data",
    "get_dbnary_data",
    "get_opendictdata",
    "get_thesaurus_data",
    # Example generation
    "generate_example_usage",
    # Dataset creation
    "create_lexical_dataset",
]
