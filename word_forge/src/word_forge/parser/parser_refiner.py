from eidosian_core import eidosian

"""Parser Refiner Module.

This module provides the core parsing and term refinement functionality for
Word Forge. It extracts lexical data from text, integrates with WordNet and
other lexical resources, and enriches word entries with relationships,
definitions, and usage examples.

Key Components:
    ParserRefiner: Main parsing pipeline that processes terms through the queue
    TermExtractor: NLP-based term discovery from text content
    ProcessingStatistics: Tracks processing metrics and statistics

The module uses NLTK for natural language processing, including lemmatization,
part-of-speech tagging, and WordNet integration. It supports concurrent
processing through a thread pool executor for efficient batch operations.

Architecture:
    Text Input → Term Extraction → Lemmatization → WordNet Lookup →
    Relationship Discovery → Database Storage → Queue Dispatch

Example:
    >>> parser = ParserRefiner(db_manager, queue_manager)
    >>> parser.process_word("algorithm")
    >>> stats = parser.get_stats()
"""

import logging  # Import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, cast

import nltk  # type: ignore
from nltk import Tree  # type: ignore
from nltk.corpus import wordnet as wn  # type: ignore
from nltk.corpus.reader.wordnet import Lemma as WordNetLemma  # type: ignore
from nltk.corpus.reader.wordnet import Synset as WordNetSynset
from nltk.stem import WordNetLemmatizer  # type: ignore

from word_forge.configs.config_essentials import LexicalDataset
from word_forge.database.database_manager import DBManager
from word_forge.parser.language_model import ModelState
from word_forge.parser.lexical_functions import create_lexical_dataset
from word_forge.queue.queue_manager import QueueManager
from word_forge.utils.nltk_utils import ensure_nltk_data

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStatistics:
    """Tracks and reports processing metrics with atomic counters."""

    processed_count: int = 0
    successful_count: int = 0
    error_count: int = 0

    @eidosian()
    def increment_processed(self) -> None:
        """Increment the processed counter."""
        self.processed_count += 1

    @eidosian()
    def increment_successful(self) -> None:
        """Increment the successful counter."""
        self.successful_count += 1

    @eidosian()
    def increment_error(self) -> None:
        """Increment the error counter."""
        self.error_count += 1

    @eidosian()
    def as_dict(self, queue_size: int, unique_words: int) -> Dict[str, int]:
        """Convert statistics to a dictionary including queue metrics."""
        return {
            "processed": self.processed_count,
            "successful": self.successful_count,
            "errors": self.error_count,
            "queue_size": queue_size,
            "unique_words": unique_words,
        }


@dataclass
class LexicalResources:
    """Manages resource paths for lexical data sources."""

    data_dir: str
    paths: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize resource paths based on data directory."""
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Define paths to all lexical resources
        self.paths = {
            "openthesaurus": f"{self.data_dir}/openthesaurus.jsonl",
            "odict": f"{self.data_dir}/odict.json",
            "dbnary": f"{self.data_dir}/dbnary.ttl",
            "opendict": f"{self.data_dir}/opendict.json",
            "thesaurus": f"{self.data_dir}/thesaurus.jsonl",
        }

    @eidosian()
    def get_path(self, resource_name: str) -> str:
        """Get the path for a specific resource."""
        return self.paths.get(resource_name, "")


class TermExtractor:
    """Discovers and extracts terms from textual content using advanced NLP techniques."""

    # Class-level flag to track whether NER is available
    _ner_available: bool = True

    def __init__(self) -> None:
        """Initialize the term extractor with necessary NLP components."""
        ensure_nltk_data()
        self._stop_words: FrozenSet[str] = frozenset(nltk.corpus.stopwords.words("english"))  # type: ignore
        self._common_words: FrozenSet[str] = frozenset(
            [
                "the",
                "and",
                "that",
                "have",
                "this",
                "with",
                "from",
                "they",
                "you",
                "what",
                "which",
                "their",
                "will",
                "would",
                "make",
                "when",
                "more",
                "other",
                "about",
                "some",
                "then",
                "than",
            ]
        )
        self._lemmatizer: WordNetLemmatizer = nltk.stem.WordNetLemmatizer()  # type: ignore

    @lru_cache(maxsize=1024)
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convert TreeBank POS tag to WordNet POS tag for accurate lemmatization.

        Args:
            treebank_tag: POS tag from NLTK's tagger

        Returns:
            WordNet POS constant for lemmatization
        """
        tag_map = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}
        return tag_map.get(treebank_tag[0], wn.NOUN)

    @eidosian()
    def extract_terms(self, definition: str, examples: List[str], original_term: str) -> Tuple[List[str], List[str]]:
        """
        Extract high-value lexical terms from definitions and examples.

        Args:
            definition: Consolidated definition text
            examples: List of usage examples
            original_term: The term being processed (to exclude from results)

        Returns:
            Tuple of (priority_terms, standard_terms) for processing
        """
        # Combine text with context markers to help NLP algorithms distinguish sources
        text_to_parse = f"DEFINITION: {definition} EXAMPLES: {' '.join(examples)}"
        original_term_lower = original_term.lower()

        # Initialize term collections
        discovered_terms: Set[str] = set()
        multiword_expressions: Set[str] = set()
        named_entities: Set[str] = set()

        # Fallback basic extraction (always performed for reliability)
        regex_terms = {word.lower() for word in re.findall(r"\b[a-zA-Z]{3,}\b", text_to_parse)}
        discovered_terms.update(regex_terms)

        try:
            # Process text with advanced NLP techniques
            sentences: List[str] = nltk.sent_tokenize(text_to_parse)  # type: ignore
            for sentence in sentences:
                self._process_sentence(sentence, discovered_terms, multiword_expressions, named_entities)

            # Extract semantically related terms via WordNet
            semantic_terms = self._extract_semantic_terms(frozenset(discovered_terms))
            discovered_terms.update(semantic_terms)

        except Exception as e:
            # Fallback to regex-only results if NLP processing fails
            logger.warning(
                f"Advanced NLP processing failed, using regex fallback: {str(e)}",
                exc_info=True,
            )

        # Filter out problematic terms
        filtered_terms = self._filter_terms(
            discovered_terms, multiword_expressions, named_entities, original_term_lower
        )

        # Score and prioritize terms
        priority_terms, standard_terms = self._score_and_sort_terms(filtered_terms)

        return priority_terms, standard_terms

    def _process_sentence(
        self,
        sentence: str,
        discovered_terms: Set[str],
        multiword_expressions: Set[str],
        named_entities: Set[str],
    ) -> None:
        """
        Process a single sentence with multiple NLP techniques.

        Args:
            sentence: Text sentence to process
            discovered_terms: Set to collect individual terms
            multiword_expressions: Set to collect multiword expressions
            named_entities: Set to collect named entities
        """
        # Basic tokenization and POS tagging
        tokens = nltk.word_tokenize(sentence)  # type: ignore
        tagged: List[Tuple[str, str]] = nltk.pos_tag(tokens)  # type: ignore

        # Extract single words with POS filtering and lemmatization
        for word, tag in tagged:
            word_lower: str = word.lower()  # Normalize to lowercase

            # Skip punctuation, short words, stop words, and numbers
            if len(word_lower) < 3 or not word_lower.isalpha() or word_lower in self._stop_words:
                continue

            # Apply proper lemmatization based on part of speech
            wordnet_pos = self._get_wordnet_pos(tag)
            lemma = self._lemmatizer.lemmatize(word_lower, wordnet_pos)
            if len(lemma) >= 3:
                discovered_terms.add(lemma)

        # Named Entity Recognition for proper nouns and terms
        self._extract_named_entities(tagged, named_entities, discovered_terms)

        # Detect useful multiword expressions
        self._extract_multiword_expressions(tagged, multiword_expressions)

    def _extract_named_entities(
        self,
        tagged: List[Tuple[str, str]],
        named_entities: Set[str],
        discovered_terms: Set[str],
    ) -> None:
        """
        Extract named entities from tagged tokens.

        Args:
            tagged: POS-tagged tokens
            named_entities: Set to add named entities to
            discovered_terms: Set to add component terms to
        """
        # Skip NER if it previously failed (e.g., missing NLTK corpus data)
        if not TermExtractor._ner_available:
            return

        try:
            chunked: Tree = nltk.ne_chunk(tagged)  # type: ignore
            for subtree in chunked:
                if isinstance(subtree, Tree) and hasattr(subtree, "label"):
                    leaves: List[Tuple[str, str]] = cast(List[Tuple[str, str]], subtree.leaves())
                    entity = " ".join(word for word, _ in leaves)
                    if len(entity) > 3:  # Filter out very short entities
                        entity_lower = entity.lower()
                        named_entities.add(entity_lower)
                        # Also add individual terms from the entity
                        for word in entity_lower.split():
                            lemma = self._lemmatizer.lemmatize(word)
                            if len(lemma) >= 3 and lemma not in self._stop_words:
                                discovered_terms.add(lemma)
        except Exception as e:
            # Soft fail for NER - disable for future calls and continue
            TermExtractor._ner_available = False
            logger.warning(
                f"Named entity recognition disabled due to error: {str(e)}. " "Processing will continue without NER."
            )

    def _extract_multiword_expressions(self, tagged: List[Tuple[str, str]], multiword_expressions: Set[str]) -> None:
        """
        Extract multiword expressions using POS patterns.

        Args:
            tagged: POS-tagged tokens
            multiword_expressions: Set to collect found expressions
        """
        if len(tagged) < 2:
            return

        # Extract phrases based on common linguistic patterns
        for i in range(len(tagged) - 1):
            # Adjective + Noun pattern (e.g., "blue sky")
            if tagged[i][1].startswith("JJ") and tagged[i + 1][1].startswith("NN"):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

            # Noun + Noun pattern (e.g., "database system")
            if tagged[i][1].startswith("NN") and tagged[i + 1][1].startswith("NN"):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

            # Verb + Particle/Adverb pattern (e.g., "log in", "set up")
            if tagged[i][1].startswith("VB") and (tagged[i + 1][1] == "RP" or tagged[i + 1][1].startswith("RB")):
                bigram = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    multiword_expressions.add(bigram)

    @lru_cache(maxsize=128)
    def _extract_semantic_terms(self, base_terms: FrozenSet[str]) -> Set[str]:
        """
        Find semantically related terms through WordNet.

        This function discovers and extracts semantically related terms by traversing
        WordNet's lexical database. It explores three relationship types to build a
        comprehensive semantic network:

        1. Synonyms - Words with the same meaning
        2. Hypernyms - Broader category terms (e.g., 'vehicle' is a hypernym of 'car')
        3. Hyponyms - More specific terms (e.g., 'sedan' is a hyponym of 'car')

        Args:
            base_terms: Initial set of discovered terms to find semantic relations for

        Returns:
            Set[str]: Collection of semantically related terms, limited to 200 results

        Note:
            The function implements performance optimizations:
            - Processes only a subset of input terms (max 75) to prevent combinatorial explosion
            - Uses LRU caching for repeated invocations with identical inputs
            - Limits result set size to 200 terms to prevent downstream overload
            - Silently continues on WordNet lookup failures (term not found, etc.)
        """
        ensure_nltk_data()
        semantic_terms: Set[str] = set()
        term_sample = list(base_terms)[:75]

        def _process_lemma(lemma: WordNetLemma) -> None:
            """Extract and normalize a lemma name, adding to results if valid."""
            lemma_name = lemma.name()
            if not isinstance(lemma_name, str):
                return

            # Normalize: replace underscores with spaces, convert to lowercase
            processed_name = lemma_name.replace("_", " ").lower()

            # Only include terms meeting minimum length requirement
            if len(processed_name) >= 3 and processed_name.replace(" ", "").isalpha():
                semantic_terms.add(processed_name)

        # Process each base term
        for base_term in term_sample:
            try:
                # Retrieve all synsets (word senses) for the term
                synsets: List[WordNetSynset] = wn.synsets(base_term)  # type: ignore
                for synset in synsets:
                    # 1. Process direct synonyms from the synset
                    for lemma in synset.lemmas():  # type: ignore
                        _process_lemma(lemma)  # type: ignore

                    # 2. Process hypernyms (broader/parent categories)
                    for hypernym in synset.hypernyms():  # type: ignore
                        for lemma in hypernym.lemmas():  # type: ignore
                            _process_lemma(lemma)  # type: ignore

                    # 3. Process hyponyms (more specific/child categories)
                    for hyponym in synset.hyponyms():  # type: ignore
                        for lemma in hyponym.lemmas():  # type: ignore
                            _process_lemma(lemma)  # type: ignore

            except (LookupError, AttributeError, ValueError, TypeError):
                continue
            except Exception as unexpected_e:
                logger.warning(
                    f"Unexpected error during WordNet lookup for '{base_term}': {unexpected_e}",
                    exc_info=True,
                )
                continue

        # Limit returned set to prevent overwhelming downstream processing
        return set(list(semantic_terms)[:200])

    def _filter_terms(
        self,
        discovered_terms: Set[str],
        multiword_expressions: Set[str],
        named_entities: Set[str],
        original_term: str,
    ) -> Set[str]:
        """
        Filter the collected terms to remove unwanted items.

        Args:
            discovered_terms: All collected terms
            multiword_expressions: Multiword expressions to preserve
            named_entities: Named entities to preserve
            original_term: Original term being processed (to exclude)

        Returns:
            Filtered set of terms
        """
        # Create combined set with all terms
        all_terms = discovered_terms.union(multiword_expressions, named_entities)

        # Remove the original term
        all_terms.discard(original_term)

        # Remove very common function words
        all_terms -= self._common_words

        return all_terms

    def _score_and_sort_terms(self, terms: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Score and sort terms by potential lexical value.

        Args:
            terms: Set of all filtered terms

        Returns:
            Tuple of (priority_terms, standard_terms)
        """
        scored_terms: List[Tuple[str, int]] = []

        for term in terms:
            # Scoring heuristics: length, complexity, multiword bonus
            score = len(term)  # Base score is length
            score += term.count(" ") * 3  # Multiword expressions get a boost
            score += sum(1 for c in term if c not in "aeiou")  # Consonant density correlates with specificity
            scored_terms.append((term, score))

        # Sort by score (higher is better)
        sorted_terms = sorted(scored_terms, key=lambda x: x[1], reverse=True)

        # Split into two categories: priority and standard
        priority_terms = [term for term, _ in sorted_terms if " " in term][:100]
        other_terms = [term for term, _ in sorted_terms if " " not in term][:150]

        return priority_terms, other_terms


class ParserRefiner:
    """
    Uses advanced lexical lookups to:
    1. Retrieve definitions, synonyms, antonyms, usage examples, etc.
    2. Store them in the DB.
    3. Enqueue new words discovered from that data for further processing.
    """

    def __init__(
        self,
        db_manager: Optional[DBManager] = None,
        queue_manager: Optional[QueueManager[str]] = None,
        data_dir: str = "data",
        model_name: Optional[str] = None,
        llm_state: Optional[ModelState] = None,
    ):
        """
        Initialize the ParserRefiner with database and queue managers.

        Args:
            db_manager: DBManager instance for database operations
            queue_manager: QueueManager instance for enqueuing new terms
            data_dir: Path to the folder containing lexical resources
            model_name: Custom model name to use (if None, uses default)
        """
        self.db_manager = db_manager or DBManager()
        self.queue_manager = queue_manager if queue_manager is not None else QueueManager[str]()
        self.resources = LexicalResources(data_dir)
        self.term_extractor = TermExtractor()
        self.stats = ProcessingStatistics()

        self.llm_state = llm_state or ModelState(model_name or "qwen/qwen2.5-0.5b-instruct")

        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=5)

    @eidosian()
    def process_word(self, term: str) -> bool:
        """
        Process a word using the integrated lexical resources.

        Data is stored in the DB and new terms are enqueued for further processing.

        Args:
            term: The word to process

        Returns:
            Boolean indicating whether processing was successful
        """
        term_lower = term.strip().lower()
        if not term_lower:
            return False

        try:
            self.stats.increment_processed()

            # Retrieve comprehensive lexical data
            dataset = create_lexical_dataset(
                term_lower,
                openthesaurus_path=self.resources.get_path("openthesaurus"),
                odict_path=self.resources.get_path("odict"),
                dbnary_path=self.resources.get_path("dbnary"),
                opendict_path=self.resources.get_path("opendict"),
                thesaurus_path=self.resources.get_path("thesaurus"),
                model_state=self.llm_state,
            )

            # Extract and consolidate word information
            definitions = self._extract_all_definitions(dataset)
            full_definition = " | ".join(definitions) if definitions else ""
            part_of_speech = self._extract_part_of_speech(dataset)
            usage_examples = self._extract_usage_examples(dataset)

            # Store in database
            self.db_manager.insert_or_update_word(
                term=term_lower,
                definition=full_definition,
                part_of_speech=part_of_speech,
                usage_examples=usage_examples,
            )

            # Process relationships and discovered terms in parallel tasks
            self._executor.submit(self._process_relationships, term_lower, dataset)
            self._executor.submit(self._discover_new_terms, term_lower, full_definition, usage_examples)

            self.stats.increment_successful()
            return True

        except Exception as e:
            self.stats.increment_error()
            logger.error(f"Error processing word '{term_lower}': {str(e)}", exc_info=True)
            return False

    def _extract_all_definitions(self, dataset: LexicalDataset) -> List[str]:
        """
        Extract and deduplicate definitions from all sources.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            List of unique definitions from all sources
        """
        combined_definitions: Set[str] = set()

        # WordNet definitions
        # Use .get with default empty list
        for wn_data in dataset.get("wordnet_data", []):
            # Ensure wn_data is a dict before accessing keys
            if isinstance(wn_data, dict):
                defn = wn_data.get("definition", "")
                if defn and isinstance(defn, str):  # Check type
                    combined_definitions.add(defn)

        # ODict / OpenDictData
        odict_data = dataset.get("odict_data", {})
        # Add type check for odict_data
        odict_def: Optional[str] = odict_data.get("definition", "") if isinstance(odict_data, dict) else None
        if odict_def and odict_def != "Not Found":
            combined_definitions.add(odict_def)

        opendict_data = dataset.get("opendict_data", {})
        # Add type check for opendict_data
        open_dict_def: Optional[str] = opendict_data.get("definition", "") if isinstance(opendict_data, dict) else None
        if open_dict_def and open_dict_def != "Not Found":
            combined_definitions.add(open_dict_def)

        # Dbnary definitions
        # Use .get with default empty list
        for item in dataset.get("dbnary_data", []):
            # Ensure item is a dict before accessing keys
            if isinstance(item, dict):
                defn = item.get("definition", "")
                if defn and isinstance(defn, str):  # Check type
                    combined_definitions.add(defn)

        return list(combined_definitions)

    def _extract_part_of_speech(self, dataset: LexicalDataset) -> str:
        """
        Extract part of speech from WordNet data if available.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            Part of speech string, or empty string if not available
        """
        # Use .get with default empty list
        wordnet_data = dataset.get("wordnet_data", [])
        # Ensure list is not empty and first element is a dict
        if wordnet_data and isinstance(wordnet_data, list) and isinstance(wordnet_data[0], dict):
            pos: Optional[str] = wordnet_data[0].get("part_of_speech", "")
            return pos if pos else ""  # Return empty string if None or empty
        return ""

    def _extract_usage_examples(self, dataset: LexicalDataset) -> List[str]:
        """
        Extract usage examples from all sources.

        Args:
            dataset: Comprehensive lexical dataset for a word

        Returns:
            List of unique usage examples from all sources
        """
        usage_examples: Set[str] = set()

        # WordNet examples
        for wn_data in dataset.get("wordnet_data", []):  # Use .get
            for ex in wn_data.get("examples", []):
                if ex:
                    usage_examples.add(ex)

        # Add auto-generated example sentence
        auto_ex = dataset.get("example_sentence", "")  # Use .get
        if auto_ex and "No example available" not in auto_ex:
            usage_examples.add(auto_ex)

        return list(usage_examples)

    def _process_relationships(self, term: str, dataset: LexicalDataset) -> None:
        """
        Process and store word relationships.

        Args:
            term: The base term
            dataset: Comprehensive lexical dataset for the term
        """
        relationship_cache: Dict[Tuple[str, str, str], bool] = {}
        discovered_terms: Set[str] = set()

        # Process WordNet relationships
        for wn_data in dataset.get("wordnet_data", []):
            for syn in wn_data.get("synonyms", []):
                syn_lower = syn.lower()
                if syn_lower != term:
                    rel_key = (term, syn_lower, "synonym")
                    if rel_key not in relationship_cache:
                        self.db_manager.insert_relationship(term, syn_lower, "synonym")
                        relationship_cache[rel_key] = True
                        discovered_terms.add(syn_lower)
            for ant in wn_data.get("antonyms", []):
                ant_lower = ant.lower()
                rel_key = (term, ant_lower, "antonym")
                if rel_key not in relationship_cache:
                    self.db_manager.insert_relationship(term, ant_lower, "antonym")
                    relationship_cache[rel_key] = True
                    discovered_terms.add(ant_lower)

        # OpenThesaurus synonyms
        for s in dataset.get("openthesaurus_synonyms", []):
            s_lower = s.lower()
            if s_lower != term:
                rel_key = (term, s_lower, "synonym")
                if rel_key not in relationship_cache:
                    self.db_manager.insert_relationship(term, s_lower, "synonym")
                    relationship_cache[rel_key] = True
                    discovered_terms.add(s_lower)

        # Thesaurus synonyms
        for s in dataset.get("thesaurus_synonyms", []):
            s_lower = s.lower()
            if s_lower != term:
                rel_key = (term, s_lower, "synonym")
                if rel_key not in relationship_cache:
                    self.db_manager.insert_relationship(term, s_lower, "synonym")
                    relationship_cache[rel_key] = True
                    discovered_terms.add(s_lower)

        # Translations from DBnary
        for item in dataset.get("dbnary_data", []):
            translation = item.get("translation", "")
            if translation:
                trans_lower = translation.lower()
                rel_key = (term, trans_lower, "translation")
                if rel_key not in relationship_cache:
                    self.db_manager.insert_relationship(term, trans_lower, "translation")
                    relationship_cache[rel_key] = True
                    discovered_terms.add(trans_lower)

        # Batch enqueue all discovered terms
        for discovered_term in discovered_terms:
            self.queue_manager.enqueue(discovered_term)

    def _discover_new_terms(self, term: str, definition: str, examples: List[str]) -> None:
        """
        Discover and enqueue new terms from definitions and examples using advanced NLP techniques.

        Args:
            term: The base term being processed
            definition: The term's consolidated definition
            examples: List of usage examples for the term
        """
        priority_terms, standard_terms = self.term_extractor.extract_terms(definition, examples, term)

        # Enqueue priority terms first (multiword expressions and specialized terms)
        for new_term in priority_terms:
            if new_term != term.lower():
                self.queue_manager.enqueue(new_term)

        # Then enqueue other discovered terms
        for new_term in standard_terms:
            if new_term != term.lower():
                self.queue_manager.enqueue(new_term)

    @eidosian()
    def get_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        queue_size = self.queue_manager.size if self.queue_manager else 0
        unique_words = 0
        if self.queue_manager and hasattr(self.queue_manager, "_seen_items"):
            try:
                unique_words = len(self.queue_manager._seen_items)  # type: ignore
            except TypeError:
                pass

        return self.stats.as_dict(
            queue_size=queue_size,
            unique_words=unique_words,
        )

    @eidosian()
    def shutdown(self) -> None:
        """Gracefully shut down resources like thread pools."""
        self._executor.shutdown(wait=True)


# Export all components for module usage
__all__ = ["ParserRefiner", "TermExtractor", "LexicalResources", "ProcessingStatistics"]
