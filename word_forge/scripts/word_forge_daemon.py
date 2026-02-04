#!/usr/bin/env python3
"""
Word Forge Daemon.

A continuous background service that:
1. Scans for incomplete words and queues them for processing.
2. Enrichs words with definitions, examples, and emotions.
3. Builds and maintains the semantic graph.
4. Continuously links terms based on vector similarity.
5. Indexes content for semantic search.

This daemon runs recursively, parallel, and asynchronously using the WorkerManager.
"""

import sys
import time
import logging
import random
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure path resolution
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.queue.queue_manager import QueueManager
from word_forge.queue.queue_worker import ParallelWordProcessor, WordProcessor, WorkerPoolConfig
from word_forge.queue.worker_manager import WorkerManager, Worker
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_worker import GraphWorker
from word_forge.vectorizer.vector_store import VectorStore
from word_forge.vectorizer.vector_worker import VectorWorker
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_worker import EmotionWorker
from word_forge.parser.parser_refiner import ParserRefiner

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("word_forge_daemon.log")
    ]
)
LOGGER = logging.getLogger("daemon")

class ScannerWorker(threading.Thread):
    """
    Scans the database for incomplete words and queues them for processing.
    Also injects seed words if the queue is empty.
    """
    def __init__(self, db_manager: DBManager, queue_manager: QueueManager, poll_interval: float = 10.0):
        super().__init__(name="ScannerWorker", daemon=True)
        self.db_manager = db_manager
        self.queue_manager = queue_manager
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self.logger = logging.getLogger("ScannerWorker")

    def run(self):
        self.logger.info("ScannerWorker started.")
        while not self._stop_event.is_set():
            try:
                # 1. Check for words needing definitions
                query = "SELECT term FROM words WHERE definition IS NULL OR definition = '' LIMIT 100"
                with self.db_manager.get_connection() as conn:
                    rows = conn.execute(query).fetchall()
                    for row in rows:
                        self.queue_manager.enqueue(row[0])
                
                # 2. If queue is empty, maybe explore neighbors of random words (Recursive Crawl)
                if self.queue_manager.is_empty:
                    self.logger.info("Queue empty. Finding recursive candidates...")
                    # Get a random word
                    with self.db_manager.get_connection() as conn:
                        row = conn.execute("SELECT term FROM words ORDER BY RANDOM() LIMIT 1").fetchone()
                        if row:
                            term = row[0]
                            # Re-queue it to force re-expansion/refresh
                            self.queue_manager.enqueue(term)
            except Exception as e:
                self.logger.error(f"Scanner error: {e}")
            
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()

    def is_alive(self) -> bool:
        return super().is_alive()

    def join(self, timeout=None):
        super().join(timeout)


class LinkerWorker(threading.Thread):
    """
    Continuously finds semantic links between terms using the Vector Store.
    """
    def __init__(self, db_manager: DBManager, vector_store: VectorStore, poll_interval: float = 5.0):
        super().__init__(name="LinkerWorker", daemon=True)
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self.logger = logging.getLogger("LinkerWorker")

    def run(self):
        self.logger.info("LinkerWorker started.")
        while not self._stop_event.is_set():
            try:
                # 1. Pick a random word from DB
                with self.db_manager.get_connection() as conn:
                    row = conn.execute("SELECT term FROM words ORDER BY RANDOM() LIMIT 1").fetchone()
                    
                if row:
                    term = row[0]
                    # 2. Search for similar terms
                    try:
                        results = self.vector_store.search(query_text=term, k=5)
                        
                        # 3. Add relationships for high similarity
                        for res in results:
                            # Search result structure: {'id': ..., 'distance': ..., 'metadata': ..., 'text': ...}
                            # Metadata has 'term'
                            meta = res.get('metadata')
                            if not meta: continue
                            
                            target_term = meta.get('term')
                            if not target_term or target_term == term:
                                continue
                            
                            # ChromaDB uses distance (lower is better), but usually cosine distance. 
                            # If it's cosine similarity, higher is better.
                            # Assuming default (L2/Cosine Distance), so smaller is better.
                            # E5/MiniLM usually uses Cosine Similarity if normalized? 
                            # Let's assume we link if it appears in top K.
                            
                            distance = res.get('distance', 1.0)
                            
                            # Add relationship to DB
                            # We use a distinct type 'semantic_similarity'
                            try:
                                self.db_manager.insert_relationship(term, target_term, "semantic_similarity")
                                # self.logger.info(f"Linked '{term}' <-> '{target_term}' (dist: {distance:.2f})")
                            except Exception:
                                pass # Likely already exists
                                
                    except Exception as e:
                         self.logger.warning(f"Linker search failed for '{term}': {e}")

            except Exception as e:
                self.logger.error(f"Linker error: {e}")
            
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()
        
    def is_alive(self) -> bool:
        return super().is_alive()
        
    def join(self, timeout=None):
        super().join(timeout)


def main():
    LOGGER.info("Initializing Word Forge Daemon...")
    
    # 1. Initialize Managers
    db_manager = DBManager()
    db_manager.create_tables()
    
    queue_manager = QueueManager()
    queue_manager.start()
    
    # Parser/Refiner (Uses LLM)
    parser_refiner = ParserRefiner(
        db_manager=db_manager,
        queue_manager=queue_manager,
        # model_name="gpt-4o" # Optional: Override if needed
    )
    
    # Processor (The core logic)
    word_processor = WordProcessor(
        db_manager=db_manager,
        parser_refiner=parser_refiner,
        logger=LOGGER
    )
    
    # Worker Pool (Parallel Processing)
    pool_config = WorkerPoolConfig(worker_count=2) # Adjust based on CPU
    processor_pool = ParallelWordProcessor(word_processor, config=pool_config, logger=LOGGER)
    
    # Graph Manager & Worker
    graph_manager = GraphManager(db_manager=db_manager)
    graph_worker = GraphWorker(graph_manager=graph_manager, poll_interval=15.0)
    
    # Vector Store & Worker
    vector_store = VectorStore(db_manager=db_manager)
    vector_worker = VectorWorker(
        db=db_manager, 
        vector_store=vector_store,
        embedder=vector_store.model_name,
        poll_interval=5.0
    )
    
    # Emotion Manager & Worker
    emotion_manager = EmotionManager(db_manager)
    emotion_worker = EmotionWorker(
        db=db_manager, 
        emotion_manager=emotion_manager, 
        poll_interval=10.0
    )
    
    # Custom Workers
    scanner_worker = ScannerWorker(db_manager, queue_manager)
    linker_worker = LinkerWorker(db_manager, vector_store)
    
    # 2. Register all with WorkerManager
    manager = WorkerManager(logger=LOGGER)
    manager.register(processor_pool)
    manager.register(graph_worker)
    manager.register(vector_worker)
    manager.register(emotion_worker)
    manager.register(scanner_worker)
    manager.register(linker_worker)
    
    # 3. Start
    LOGGER.info("Starting all workers...")
    manager.start_all()
    
    # Seed if empty
    if queue_manager.is_empty:
        LOGGER.info("Seeding queue...")
        seeds = ["system", "complexity", "emergence", "recursive", "optimization"]
        for s in seeds:
            queue_manager.enqueue(s)
            
    # 4. Monitor Loop
    try:
        while True:
            time.sleep(5)
            # Log status
            q_size = queue_manager.size()
            pool_status = processor_pool.get_status()
            
            LOGGER.info(f"--- Daemon Status ---")
            LOGGER.info(f"Queue Size: {q_size}")
            LOGGER.info(f"Processed: {pool_status['stats'].get('processed_count', 0)}")
            LOGGER.info(f"Graph: {graph_worker.get_status()['state']}")
            
            # Simple health check
            if not manager.any_alive():
                LOGGER.error("All workers died!")
                break
                
    except KeyboardInterrupt:
        LOGGER.info("Stopping daemon...")
    finally:
        manager.stop_all()
        queue_manager.stop()
        db_manager.close()
        LOGGER.info("Daemon stopped.")

if __name__ == "__main__":
    main()
