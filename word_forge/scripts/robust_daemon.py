#!/usr/bin/env python3
"""
Word Forge Robust Daemon.

A production-grade background service implementing:
- Distributed-style Task Queuing (SQLite-backed persistence implicit in design pattern).
- Robust Error Handling & Retry Logic.
- Recursive Enrichment Loop (Saturation-based).
- Monitoring & Metrics Endpoint (File-based heartbeat).
- Periodic Visualization Trigger (Every 100 complete entries).
- Comprehensive Logging.

This daemon runs indefinitely until manually stopped or task saturation.
"""

import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure path resolution
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

from word_forge.database import TermNotFoundError
from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.multilingual import ingest_kaikki_jsonl, ingest_wiktextract_jsonl
from word_forge.multilingual.multilingual_manager import MultilingualManager
from word_forge.parser.language_model import ModelState
from word_forge.parser.lexical_functions import create_lexical_dataset, generate_comprehensive_enrichment
from word_forge.queue.queue_manager import QueueManager, TaskPriority
from word_forge.vectorizer.vector_store import VectorStore

# Configuration
LOG_FILE = "word_forge_daemon.log"
STATUS_FILE = "daemon_status.json"
VIZ_DIR = Path("visualizations")
BATCH_SIZE = 100

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)
LOGGER = logging.getLogger("Daemon")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _split_definitions(definition: str) -> List[str]:
    if not definition:
        return []
    parts = [p.strip() for p in definition.split(" | ") if p.strip()]
    if not parts:
        parts = [p.strip() for p in definition.split("\n") if p.strip()]
    return parts


def _merge_unique(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for item in list(existing) + list(incoming):
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _merge_entry_fields(
    existing_definition: str,
    existing_examples: List[str],
    new_definitions: List[str],
    new_examples: List[str],
) -> Tuple[str, List[str]]:
    merged_definitions = _merge_unique(
        _split_definitions(existing_definition),
        new_definitions,
    )
    merged_examples = _merge_unique(existing_examples, new_examples)
    return " | ".join(merged_definitions), merged_examples


class Monitor:
    """Manages health checks and status reporting."""

    def __init__(self, status_file: str):
        self.status_file = status_file
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.last_viz_count = 0
        self.node_count = 0
        self.llm_completed = 0
        self.last_viz_node_count = 0
        self.last_viz_llm_count = 0
        self.multilingual_processed = 0
        self.queue_size = 0
        self.last_term = "None"
        self._lock = threading.Lock()
        # Write initial state
        self.update(status="INITIALIZING")

    def update(
        self,
        processed_delta=0,
        error_delta=0,
        queue_size=0,
        node_delta=0,
        llm_delta=0,
        multilingual_delta=0,
        status="RUNNING",
        last_term=None,
    ):
        with self._lock:
            self.processed_count += processed_delta
            self.error_count += error_delta
            self.node_count += node_delta
            self.llm_completed += llm_delta
            self.multilingual_processed += multilingual_delta
            self.queue_size = queue_size
            if last_term:
                self.last_term = last_term
            self._write_status(status)

    def _write_status(self, status_str):
        status_data = {
            "status": status_str,
            "uptime_seconds": time.time() - self.start_time,
            "processed_items": self.processed_count,
            "errors": self.error_count,
            "graph_nodes": self.node_count,
            "llm_completed": self.llm_completed,
            "multilingual_processed": self.multilingual_processed,
            "queue_size": self.queue_size,
            "last_term": self.last_term,
            "last_heartbeat": datetime.now().isoformat(),
            "model": "qwen/qwen2.5-1.5b-instruct",
        }
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)


class LexicalWorker(threading.Thread):
    """
    Non-LLM lexical ingestion worker. Keeps the pipeline moving while LLM fills trickle in.
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        llm_queue: QueueManager,
        graph_queue: QueueManager,
        vector_queue: QueueManager,
        db_manager: DBManager,
        monitor: Monitor,
    ):
        super().__init__(name="LexicalWorker", daemon=True)
        self.queue = queue_manager
        self.llm_queue = llm_queue
        self.graph_queue = graph_queue
        self.vector_queue = vector_queue
        self.db = db_manager
        self.monitor = monitor
        self._stop_event = threading.Event()

    def _entry_needs_llm(self, entry: Dict[str, Any]) -> bool:
        definition = (entry.get("definition") or "").strip()
        examples = entry.get("usage_examples") or []
        if isinstance(examples, str):
            examples = [examples]
        has_examples = any(isinstance(x, str) and x.strip() for x in examples)
        return not definition or not has_examples

    def run(self):
        LOGGER.info("LexicalWorker started (non-LLM).")

        while not self._stop_event.is_set():
            if self.queue.is_empty:
                time.sleep(2)
                continue

            try:
                res = self.queue.dequeue()
                if res.is_failure:
                    continue

                term = res.unwrap()
                LOGGER.info(f"Processing (lexical): {term}")
                self.monitor.update(last_term=term, queue_size=self.queue.size)

                dataset = create_lexical_dataset(
                    term,
                    model_state=None,
                )

                definitions = []
                with contextlib.suppress(Exception):
                    from word_forge.parser.parser_refiner import ParserRefiner

                    parser = ParserRefiner(db_manager=self.db)
                    definitions = parser._extract_all_definitions(dataset)
                    part_of_speech = parser._extract_part_of_speech(dataset)
                    usage_examples = parser._extract_usage_examples(dataset)
                if not definitions:
                    part_of_speech = "noun"
                    usage_examples = []

                full_definition = " | ".join(definitions) if definitions else ""

                try:
                    existing_entry = self.db.get_word_entry(term)
                    merged_definition, merged_examples = _merge_entry_fields(
                        existing_entry.get("definition", ""),
                        existing_entry.get("usage_examples", []) or [],
                        definitions,
                        usage_examples,
                    )
                except TermNotFoundError:
                    merged_definition = full_definition
                    merged_examples = usage_examples

                self.db.insert_or_update_word(
                    term=term,
                    definition=merged_definition,
                    part_of_speech=part_of_speech,
                    usage_examples=merged_examples,
                )

                # Discover additional terms and phrases from dataset/definitions/examples
                discovered_terms = self._discover_terms_from_dataset(
                    dataset,
                    definitions,
                    usage_examples,
                )
                for new_term in discovered_terms:
                    if new_term != term:
                        self.queue.enqueue(new_term, priority=TaskPriority.LOW)

                try:
                    entry = self.db.get_word_entry(term)
                    if self._entry_needs_llm(entry):
                        self.llm_queue.enqueue(term, priority=TaskPriority.LOW)
                except TermNotFoundError:
                    self.llm_queue.enqueue(term, priority=TaskPriority.NORMAL)

                self.graph_queue.enqueue(term, priority=TaskPriority.LOW)
                self.vector_queue.enqueue(term, priority=TaskPriority.LOW)

                self.monitor.update(processed_delta=1, queue_size=self.queue.size)

            except Exception as e:
                LOGGER.error(f"LexicalWorker error for '{term}': {e}")
                self.monitor.update(error_delta=1)
                with contextlib.suppress(Exception):
                    self.queue.enqueue(term, priority=TaskPriority.LOW)

    def stop(self):
        self._stop_event.set()

    def _discover_terms_from_dataset(
        self,
        dataset: Dict[str, Any],
        definitions: List[str],
        usage_examples: List[str],
    ) -> List[str]:
        """Extract candidate terms/phrases for recursive expansion."""
        candidates: List[str] = []

        def _add(value: Any) -> None:
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        candidates.append(item)

        _add(dataset.get("openthesaurus_synonyms", []))
        _add(dataset.get("thesaurus_synonyms", []))

        for wn_entry in dataset.get("wordnet_data", []):
            if not isinstance(wn_entry, dict):
                continue
            _add(wn_entry.get("synonyms", []))
            _add(wn_entry.get("antonyms", []))
            _add(wn_entry.get("hypernyms", []))
            _add(wn_entry.get("hyponyms", []))
            _add(wn_entry.get("holonyms", []))
            _add(wn_entry.get("meronyms", []))
            _add(wn_entry.get("examples", []))

        _add(definitions)
        _add(usage_examples)

        normalized: List[str] = []
        for raw in candidates:
            cleaned = raw.strip().lower()
            if not cleaned:
                continue
            # Extract short phrases from longer strings
            if len(cleaned.split()) > 6:
                continue
            if any(ch.isdigit() for ch in cleaned):
                continue
            if len(cleaned) < 2 or len(cleaned) > 48:
                continue
            normalized.append(cleaned)

        # Simple phrase extraction: split on punctuation and filter
        phrases: List[str] = []
        for item in normalized:
            parts = re.split(r"[;,.()\[\]:]+", item)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part in _STOPWORDS:
                    continue
                if 2 <= len(part.split()) <= 4:
                    phrases.append(part)
                elif len(part.split()) == 1:
                    if part not in _STOPWORDS:
                        phrases.append(part)

        # Deduplicate
        seen = set()
        results: List[str] = []
        for term in phrases:
            if term not in seen:
                seen.add(term)
                results.append(term)
        return results


class GraphUpdateWorker(threading.Thread):
    """Continuously updates the graph without blocking lexical or LLM pipelines."""

    def __init__(
        self,
        graph_queue: QueueManager,
        db_manager: DBManager,
        graph_manager: GraphManager,
        monitor: Monitor,
    ):
        super().__init__(name="GraphUpdateWorker", daemon=True)
        self.queue = graph_queue
        self.db = db_manager
        self.graph = graph_manager
        self.monitor = monitor
        self._stop_event = threading.Event()

    def run(self):
        LOGGER.info("GraphUpdateWorker started.")
        while not self._stop_event.is_set():
            if self.queue.is_empty:
                time.sleep(2)
                continue

            try:
                res = self.queue.dequeue()
                if res.is_failure:
                    continue
                term = res.unwrap()
                try:
                    entry = self.db.get_word_entry(term)
                except TermNotFoundError:
                    continue
                with self.db.get_connection() as conn:
                    lang_count = conn.execute(
                        "SELECT COUNT(DISTINCT lang) FROM lexemes WHERE base_term = ?",
                        (term,),
                    ).fetchone()
                entry["lang_count"] = int(lang_count[0]) if lang_count else 0
                entry["definition_count"] = len(_split_definitions(entry.get("definition", "")))
                examples = entry.get("usage_examples") or []
                entry["example_count"] = len(examples) if isinstance(examples, list) else 0

                term_key = term.lower()
                with self.graph._graph_lock:
                    existed = term_key in self.graph._term_to_id
                node_id = self.graph.add_word_node(term, attributes=entry)
                entry["degree"] = self.graph.g.degree(node_id)
                if not existed:
                    self.monitor.update(node_delta=1)
                    if self.monitor.node_count - self.monitor.last_viz_node_count >= 100:
                        self._trigger_visualization()
                        self.monitor.last_viz_node_count = self.monitor.node_count
            except Exception as e:
                LOGGER.error(f"Graph update failed: {e}")

    def stop(self):
        self._stop_event.set()

    def _trigger_visualization(self):
        LOGGER.info("Generating Batch Visualization (nodes)...")
        try:
            VIZ_DIR.mkdir(exist_ok=True)
            timestamp = int(time.time())
            self.graph.visualize(
                output_path=str(VIZ_DIR / f"graph_2d_{timestamp}.html"),
                use_3d=False,
            )
            self.graph.visualize(
                output_path=str(VIZ_DIR / f"graph_3d_{timestamp}.html"),
                use_3d=True,
            )
        except Exception as e:
            LOGGER.error(f"Visualization failed: {e}")


class VectorUpdateWorker(threading.Thread):
    """Continuously updates vector store without blocking lexical or LLM pipelines."""

    def __init__(
        self,
        vector_queue: QueueManager,
        db_manager: DBManager,
        vector_store: VectorStore,
    ):
        super().__init__(name="VectorUpdateWorker", daemon=True)
        self.queue = vector_queue
        self.db = db_manager
        self.vectors = vector_store
        self._stop_event = threading.Event()

    def run(self):
        LOGGER.info("VectorUpdateWorker started.")
        while not self._stop_event.is_set():
            if self.queue.is_empty:
                time.sleep(2)
                continue

            try:
                res = self.queue.dequeue()
                if res.is_failure:
                    continue
                term = res.unwrap()
                try:
                    entry = self.db.get_word_entry(term)
                    self.vectors.store_word(entry)
                except TermNotFoundError:
                    continue
            except Exception as e:
                LOGGER.error(f"Vector update failed: {e}")

    def stop(self):
        self._stop_event.set()


class MultilingualWorker(threading.Thread):
    """Ingest multilingual JSONL data and align to English base terms."""

    def __init__(
        self,
        multilingual_queue: QueueManager,
        lexical_queue: QueueManager,
        graph_queue: QueueManager,
        vector_queue: QueueManager,
        monitor: Monitor,
        manager: Optional[MultilingualManager] = None,
    ):
        super().__init__(name="MultilingualWorker", daemon=True)
        self.queue = multilingual_queue
        self.lexical_queue = lexical_queue
        self.graph_queue = graph_queue
        self.vector_queue = vector_queue
        self.monitor = monitor
        self.manager = manager or MultilingualManager()
        self._stop_event = threading.Event()

    def run(self):
        LOGGER.info("MultilingualWorker started.")
        while not self._stop_event.is_set():
            if self.queue.is_empty:
                time.sleep(2)
                continue

            try:
                res = self.queue.dequeue()
                if res.is_failure:
                    continue
                task = res.unwrap()

                if isinstance(task, dict):
                    path = str(task.get("path") or "").strip()
                    source = str(task.get("source") or "").strip().lower()
                    base_lang = str(task.get("base_lang") or "en").strip()
                    limit = task.get("limit")
                else:
                    path = str(task)
                    source = "wiktextract"
                    base_lang = "en"
                    limit = None

                if not path:
                    continue

                if source == "kaikki":
                    ingest_kaikki_jsonl(path, manager=self.manager, base_lang=base_lang, limit=limit)
                else:
                    ingest_wiktextract_jsonl(path, manager=self.manager, base_lang=base_lang, limit=limit)

                # enqueue base terms discovered during ingestion
                with self.manager.db.get_connection() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT base_term FROM lexemes WHERE base_term IS NOT NULL AND base_term != ''"
                    ).fetchall()
                    for row in rows:
                        base_term = row[0]
                        self.lexical_queue.enqueue(base_term, priority=TaskPriority.LOW)
                        self.graph_queue.enqueue(base_term, priority=TaskPriority.LOW)
                        self.vector_queue.enqueue(base_term, priority=TaskPriority.LOW)

                self.monitor.update(multilingual_delta=1)
            except Exception as e:
                LOGGER.error(f"Multilingual ingestion failed: {e}")
                self.monitor.update(error_delta=1)

    def stop(self):
        self._stop_event.set()


class Scanner(threading.Thread):
    """Feeds the queue from DB or seeds."""

    def __init__(self, db: DBManager, queue: QueueManager):
        super().__init__(name="Scanner", daemon=True)
        self.db = db
        self.queue = queue
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            if self.queue.is_empty:
                # Find terms with few relationships or no definition
                # Or inject seeds
                seeds = ["consciousness", "entropy", "recursion", "harmony"]
                for s in seeds:
                    self.queue.enqueue(s, priority=TaskPriority.NORMAL)
            time.sleep(10)  # Scan interval

    def stop(self):
        self._stop_event.set()


class LLMFillWorker(threading.Thread):
    """
    LLM-only enrichment worker that fills in incomplete entries.
    """

    def __init__(
        self,
        llm_queue: QueueManager,
        graph_queue: QueueManager,
        vector_queue: QueueManager,
        db_manager: DBManager,
        graph_manager: GraphManager,
        monitor: Monitor,
    ):
        super().__init__(name="LLMFillWorker", daemon=True)
        self.queue = llm_queue
        self.graph_queue = graph_queue
        self.vector_queue = vector_queue
        self.db = db_manager
        self.graph = graph_manager
        self.monitor = monitor
        self.model_state = ModelState(model_name="ollama:qwen2.5:1.5b-Instruct")
        self._stop_event = threading.Event()

    def _entry_complete(self, entry: Dict[str, Any]) -> bool:
        definition = (entry.get("definition") or "").strip()
        examples = entry.get("usage_examples") or []
        if isinstance(examples, str):
            examples = [examples]
        has_examples = any(isinstance(x, str) and x.strip() for x in examples)
        return bool(definition and has_examples)

    def run(self):
        LOGGER.info("LLMFillWorker started with qwen/qwen2.5-1.5b-instruct")
        if not self.model_state.initialize():
            LOGGER.error("LLM model initialization failed. LLMFillWorker stopping.")
            return

        while not self._stop_event.is_set():
            if self.queue.is_empty:
                time.sleep(2)
                continue

            try:
                res = self.queue.dequeue()
                if res.is_failure:
                    continue

                term = res.unwrap()
                try:
                    entry = self.db.get_word_entry(term)
                except TermNotFoundError:
                    self.queue.enqueue(term, priority=TaskPriority.NORMAL)
                    continue

                if self._entry_complete(entry):
                    continue

                definition = (entry.get("definition") or "").strip()
                pos = (entry.get("part_of_speech") or "noun").strip()

                enrichment = generate_comprehensive_enrichment(
                    word=term,
                    definition=definition,
                    pos=pos,
                    model_state=self.model_state,
                )

                new_definitions = []
                if isinstance(enrichment.get("definition"), str):
                    new_definitions = [enrichment["definition"]]
                new_examples = enrichment.get("usage_examples") or []
                if isinstance(new_examples, str):
                    new_examples = [new_examples]

                merged_definition, merged_examples = _merge_entry_fields(
                    entry.get("definition", ""),
                    entry.get("usage_examples", []) or [],
                    new_definitions,
                    new_examples,
                )
                self.db.insert_or_update_word(
                    term=term,
                    definition=merged_definition,
                    part_of_speech=pos,
                    usage_examples=merged_examples,
                )

                for rel_type in ["holonyms", "meronyms", "hypernyms", "hyponyms"]:
                    if rel_type in enrichment:
                        for target in enrichment[rel_type]:
                            if isinstance(target, str):
                                self.db.insert_or_update_word(target)
                                self.db.insert_relationship(term, target, rel_type[:-1])

                self.graph_queue.enqueue(term, priority=TaskPriority.LOW)
                self.vector_queue.enqueue(term, priority=TaskPriority.LOW)
                # Track completion and trigger visualization on completed batches
                updated_entry = self.db.get_word_entry(term)
                if self._entry_complete(updated_entry):
                    self.monitor.update(llm_delta=1)
                    if self.monitor.llm_completed - self.monitor.last_viz_llm_count >= 100:
                        self._trigger_visualization()
                        self.monitor.last_viz_llm_count = self.monitor.llm_completed
                self.monitor.update(processed_delta=1)

            except Exception as e:
                LOGGER.error(f"Error in LLM fill for '{term}': {e}")
                self.monitor.update(error_delta=1)
                with contextlib.suppress(Exception):
                    self.queue.enqueue(term, priority=TaskPriority.LOW)

    def stop(self):
        self._stop_event.set()

    def _trigger_visualization(self):
        LOGGER.info("Generating Batch Visualization (LLM completed)...")
        try:
            VIZ_DIR.mkdir(exist_ok=True)
            timestamp = int(time.time())
            self.graph.visualize(
                output_path=str(VIZ_DIR / f"graph_2d_{timestamp}.html"),
                use_3d=False,
            )
            self.graph.visualize(
                output_path=str(VIZ_DIR / f"graph_3d_{timestamp}.html"),
                use_3d=True,
            )
        except Exception as e:
            LOGGER.error(f"Visualization trigger failed: {e}")


def main():
    LOGGER.info("Starting Word Forge Robust Daemon...")

    # Init Components
    db = DBManager()
    db.create_tables()
    qm = QueueManager()
    llm_qm = QueueManager()
    graph_qm = QueueManager()
    vector_qm = QueueManager()
    multilingual_qm = QueueManager()
    gm = GraphManager(db_manager=db)
    vs = VectorStore(
        db_manager=db,
        model_name="ollama:nomic-embed-text",
        collection_name="word_forge_vectors_ollama_768",
    )
    monitor = Monitor(STATUS_FILE)

    # Workers
    worker = LexicalWorker(qm, llm_qm, graph_qm, vector_qm, db, monitor)
    graph_worker = GraphUpdateWorker(graph_qm, db, gm, monitor)
    vector_worker = VectorUpdateWorker(vector_qm, db, vs)
    llm_worker = LLMFillWorker(llm_qm, graph_qm, vector_qm, db, gm, monitor)
    multilingual_worker = MultilingualWorker(
        multilingual_qm,
        qm,
        graph_qm,
        vector_qm,
        monitor,
    )
    scanner = Scanner(db, qm)

    worker.start()
    graph_worker.start()
    vector_worker.start()
    llm_worker.start()
    multilingual_worker.start()
    scanner.start()

    multilingual_dir = os.environ.get("WORD_FORGE_MULTILINGUAL_DIR", "").strip()
    if multilingual_dir:
        for path in Path(multilingual_dir).glob("*.jsonl"):
            source = "kaikki" if "kaikki" in path.name.lower() else "wiktextract"
            multilingual_qm.enqueue({"path": str(path), "source": source, "base_lang": "en"})

    LOGGER.info(f"Daemon running. Check {STATUS_FILE} for status.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Stopping...")
        worker.stop()
        graph_worker.stop()
        vector_worker.stop()
        llm_worker.stop()
        multilingual_worker.stop()
        scanner.stop()
        worker.join()
        graph_worker.join()
        vector_worker.join()
        llm_worker.join()
        multilingual_worker.join()
        scanner.join()
        LOGGER.info("Stopped.")


if __name__ == "__main__":
    main()
