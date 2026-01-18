import os
import signal
import pickle
import json
import shutil
import logging
import requests
import threading
import argparse
import hashlib
import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Union, Any
import graphrag  # New import
import numpy as np
import networkx as nx
from tqdm import tqdm
import concurrent.futures

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("semantic_fs.log"),
        logging.StreamHandler()
    ]
)


class Config:
    """Centralized configuration class for global settings."""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    GRAPH_DB_PATH: str = './graph_db.pkl'
    LINK_FS_ROOT: str = './semantic_fs/'
    CONTEXTUAL_SEARCH_K: int = 10
    LOCAL_LLM_ENDPOINT: str = 'http://localhost:11434/api/generate'
    LOCAL_LLM_MODEL: str = 'deepseek-r1:1.5b'
    MAX_WORKERS: int = 4
    MAX_TEXT_SIZE_MB: int = 5
    FILE_BATCH_SIZE: int = 100
    SUPPORTED_TEXT_TYPES: Set[str] = {
        'text/plain', 'text/html', 'text/css', 'text/javascript',
        'application/json', 'application/xml', 'application/csv',
        'application/x-python', 'application/x-java-source'
    }
    EXCLUDED_DIRS: Set[str] = {
        '.git', 'node_modules', 'venv', '__pycache__', '.idea',
        '.vscode', 'build', 'dist', 'target'
    }
    EXCLUDED_FILES: Set[str] = {
        '.DS_Store', 'Thumbs.db', '.gitignore'
    }


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Calculate MD5 hash of file content for change detection."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    @staticmethod
    def is_text_file(filepath: str) -> bool:
        """Determine if a file is text and can be embedded."""
        # Check file size first
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > Config.MAX_TEXT_SIZE_MB:
                return False
                
            # Check file type
            mime_type, _ = mimetypes.guess_type(filepath)
            if mime_type and mime_type in Config.SUPPORTED_TEXT_TYPES:
                return True
                
            # Try reading first few lines
            with open(filepath, 'r', errors='ignore') as f:
                try:
                    f.read(4096)
                    return True
                except UnicodeDecodeError:
                    return False
        except Exception:
            return False
    
    @staticmethod
    def should_process_file(filepath: str) -> bool:
        """Determine if a file should be processed."""
        # Check if file is in excluded list
        filename = os.path.basename(filepath)
        if filename in Config.EXCLUDED_FILES:
            return False
            
        # Check if in excluded directory
        for excluded_dir in Config.EXCLUDED_DIRS:
            if excluded_dir in filepath.split(os.path.sep):
                return False
                
        # Check if it's a text file we can process
        return FileUtils.is_text_file(filepath)


class SemanticFileSystem:
    """Generates semantic filesystem structure based on knowledge graph."""

    def __init__(self, graph: nx.Graph, llm: Optional['LocalLLM'] = None) -> None:
        self.graph = graph
        self.llm = llm

    def generate(self) -> None:
        """Generate the semantic filesystem with symlinks."""
        # Clean existing structure
        if os.path.exists(Config.LINK_FS_ROOT):
            shutil.rmtree(Config.LINK_FS_ROOT)
        os.makedirs(Config.LINK_FS_ROOT, exist_ok=True)
        
        # Create structure based on categories and relationships
        self._create_category_dirs()
        self._create_relationship_dirs()
        logging.info(f"Semantic filesystem structure generated at {Config.LINK_FS_ROOT}")
        
    def _create_category_dirs(self) -> None:
        """Create directories based on file categories."""
        for node in self.graph.nodes:
            metadata = self.graph.nodes[node]['metadata']
            category = metadata.get('category', 'uncategorized')
            
            # Create category directory
            dir_path = Path(Config.LINK_FS_ROOT) / 'by_category' / category
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create symlink if file exists
            if os.path.exists(node):
                target_path = dir_path / Path(node).name
                if not target_path.exists():
                    os.symlink(os.path.abspath(node), target_path)
                    
    def _create_relationship_dirs(self) -> None:
        """Create directories based on document relationships."""
        # Create clusters directory
        clusters_dir = Path(Config.LINK_FS_ROOT) / 'by_cluster'
        clusters_dir.mkdir(parents=True, exist_ok=True)
        
        # Find connected components (clusters of related documents)
        for i, component in enumerate(nx.connected_components(self.graph)):
            if len(component) < 2:  # Skip isolated nodes
                continue
                
            # Create cluster directory
            cluster_dir = clusters_dir / f"cluster_{i}"
            cluster_dir.mkdir(exist_ok=True)
            
            # Create topic label using LLM if available
            if self.llm:
                filepaths = list(component)[:5]  # Take first 5 files to avoid overloading LLM
                file_contents = []
                for path in filepaths:
                    try:
                        with open(path, 'r', errors='ignore') as f:
                            file_contents.append(f.read()[:1000])  # Use first 1000 chars
                    except Exception:
                        pass
                
                if file_contents:
                    prompt = f"These files seem related. Generate a concise 2-3 word topic label:\n{file_contents}"
                    topic = self.llm.query(prompt)
                    if topic:
                        # Sanitize topic name for filesystem
                        topic = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in topic)
                        topic = topic.strip()[:50]
                        # Create symlink to cluster with topic name
                        topic_link = clusters_dir / topic
                        if not topic_link.exists():
                            os.symlink(cluster_dir, topic_link)
            
            # Create symlinks for files in cluster
            for filepath in component:
                if os.path.exists(filepath):
                    target_path = cluster_dir / Path(filepath).name
                    if not target_path.exists():
                        os.symlink(os.path.abspath(filepath), target_path)


class LocalLLM:
    """Interfaces with a locally hosted LLM for intelligent queries."""

    def __init__(self, model_name: str = Config.LOCAL_LLM_MODEL) -> None:
        self.model_name = model_name
        self._available = self._check_availability()
        logging.info(f"Local LLM initialized with model: {model_name} (available: {self._available})")

    def _check_availability(self) -> bool:
        """Check if the LLM endpoint is available."""
        try:
            requests.get(Config.LOCAL_LLM_ENDPOINT.split('/api')[0], timeout=1)
            return True
        except Exception:
            logging.warning("Local LLM endpoint not available. Some features will be limited.")
            return False

    def query(self, prompt: str) -> Optional[str]:
        """Query the LLM with a prompt."""
        if not self._available:
            return None
            
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'max_tokens': 500
        }
        try:
            response = requests.post(Config.LOCAL_LLM_ENDPOINT, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()['response']
            logging.debug(f"LLM query successful for prompt: '{prompt[:30]}...'")
            return result
        except requests.RequestException as e:
            logging.error(f"LLM query failed: {e}")
            return None
            
    def categorize_document(self, content: str, filepath: str) -> Dict[str, Any]:
        """Use LLM to categorize document content."""
        if not self._available:
            # Default categorization based on file extension
            ext = os.path.splitext(filepath)[1].lower()
            category_map = {
                '.py': 'python', '.js': 'javascript', '.html': 'web',
                '.css': 'web', '.md': 'documentation', '.txt': 'text',
                '.json': 'data', '.csv': 'data', '.xml': 'data'
            }
            return {
                'category': category_map.get(ext, 'general'),
                'auto_categorized': True
            }
            
        # Prepare snippet (first and last bit of content)
        if len(content) > 2000:
            snippet = content[:1000] + "\n...\n" + content[-1000:]
        else:
            snippet = content
            
        prompt = f"""Analyze this file and determine:
1. The main category (one word): programming, data, document, config, etc.
2. The primary programming language or markup language (if applicable)
3. The general topic (2-3 words)
4. Keywords (3-5 comma separated)

File path: {filepath}
Content snippet:
{snippet}

Respond in JSON format only:
"""
        
        try:
            response = self.query(prompt)
            if response:
                # Clean up response to get only JSON part
                json_part = response.strip()
                if json_part.startswith("```json"):
                    json_part = json_part.split("```json")[1]
                if json_part.endswith("```"):
                    json_part = json_part.split("```")[0]
                    
                metadata = json.loads(json_part)
                metadata['auto_categorized'] = True
                return metadata
        except Exception as e:
            logging.error(f"Failed to categorize document with LLM: {e}")
            
        # Fallback
        return {'category': 'general', 'auto_categorized': True}


class IntelligentFileSystem:
    """Main orchestrator class managing embedding, indexing, querying, and filesystem generation."""

    def __init__(self) -> None:
        self.llm = LocalLLM()
        self.graph = nx.Graph()
        self.semantic_fs = SemanticFileSystem(self.graph, self.llm)
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.shutdown_event = threading.Event()
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None
        }

    def index_directory(self, directory: str, incremental: bool = True) -> None:
        """Index all suitable files in a directory."""
        logging.info(f"Starting indexing of directory: {directory}")
        self.stats['start_time'] = time.time()
        self.stats['processed'] = 0
        self.stats['failed'] = 0  
        self.stats['skipped'] = 0
        
        # Get all files to process
        all_files = []
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in Config.EXCLUDED_DIRS]
            
            for file in files:
                filepath = os.path.join(root, file)
                if FileUtils.should_process_file(filepath):
                    # Check if incremental and file hasn't changed
                    if incremental and not self.has_changed(filepath):
                        self.stats['skipped'] += 1
                        continue
                    all_files.append(filepath)
        
        # Process files with progress bar
        with tqdm(total=len(all_files), desc="Indexing files") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                # Submit file indexing tasks in batches
                for i in range(0, len(all_files), Config.FILE_BATCH_SIZE):
                    batch = all_files[i:i+Config.FILE_BATCH_SIZE]
                    futures = [executor.submit(self._process_file, filepath) for filepath in batch]
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result == "processed":
                            self.stats['processed'] += 1
                        elif result == "failed":
                            self.stats['failed'] += 1
                        pbar.update(1)
                        
                        # Check shutdown event
                        if self.shutdown_event.is_set():
                            logging.warning("Shutdown detected. Halting indexing.")
                            return
            
        duration = time.time() - self.stats['start_time']
        logging.info(f"Indexing complete. Processed: {self.stats['processed']}, "
                    f"Failed: {self.stats['failed']}, Skipped: {self.stats['skipped']} "
                    f"in {duration:.1f} seconds")
        self.save()

    def _process_file(self, filepath: str) -> str:
        """Process a single file for indexing."""
        try:
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
                
            # Create embedding and add to graph using GraphRAG
            graphrag.index(
                root_dir=filepath,
                model="deepseek-r1:1.5b",  # or "ollama" as configured
                # ...other GraphRAG configs...
            )
            
            # Get metadata including LLM categorization
            metadata = self.llm.categorize_document(content, filepath)
            
            # Add file info
            file_stat = os.stat(filepath)
            metadata.update({
                'size': file_stat.st_size,
                'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'file_type': mimetypes.guess_type(filepath)[0] or 'unknown'
            })
            
            # Add to graph
            self.graph.add_node(filepath, metadata=metadata)
            return "processed"
        except Exception as e:
            logging.error(f"Failed to index {filepath}: {e}")
            return "failed"

    def semantic_search(self, query: str, top_k: int = Config.CONTEXTUAL_SEARCH_K,
                      filter_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for documents semantically similar to the query."""
        results = graphrag.query(
            query=query,
            method="local",  # or "global"/"drift"
            # ...other GraphRAG configs...
        )
        
        # Format and filter results
        formatted_results = []
        for result in results:
            filepath = result['filepath']
            metadata = self.graph.nodes[filepath]['metadata']
            
            # Apply category filter if specified
            if filter_category and metadata.get('category') != filter_category:
                continue
                
            result = {
                'filepath': filepath,
                'score': float(result['score']),
                'metadata': metadata,
                'filename': os.path.basename(filepath)
            }
            formatted_results.append(result)
            
            # Stop after we have enough results
            if len(formatted_results) >= top_k:
                break
                
        return formatted_results

    def regenerate_file_structure(self) -> None:
        """Generate semantic filesystem structure."""
        logging.info("Regenerating semantic filesystem structure.")
        self.semantic_fs.generate()

    def cleanup(self) -> int:
        """Remove entries for files that no longer exist."""
        logging.info("Cleaning up missing files from index...")
        removed_count = 0
        
        for filepath in list(self.graph.nodes):
            if not os.path.exists(filepath):
                logging.debug(f"Removing {filepath} - file no longer exists")
                self.graph.remove_node(filepath)
                removed_count += 1
                    
        if removed_count > 0:
            self.save()
            logging.info(f"Removed {removed_count} missing files from index")
        return removed_count

    def has_changed(self, filepath: str) -> bool:
        """Check if file has changed since last indexing."""
        current_hash = FileUtils.get_file_hash(filepath)
        return current_hash != self.graph.nodes[filepath]['metadata'].get('file_hash', "")

    def save(self) -> None:
        """Save graph to disk."""
        with open(Config.GRAPH_DB_PATH, 'wb') as f:
            pickle.dump(self.graph, f)
        logging.info(f"Knowledge graph saved with {len(self.graph.nodes)} nodes.")

    def load(self) -> None:
        """Load graph from disk."""
        if Path(Config.GRAPH_DB_PATH).exists():
            try:
                with open(Config.GRAPH_DB_PATH, 'rb') as f:
                    self.graph = pickle.load(f)
                logging.info(f"Knowledge graph loaded with {len(self.graph.nodes)} documents.")
            except Exception as e:
                logging.error(f"Failed to load knowledge graph: {e}")
                self.graph = nx.Graph()
        else:
            logging.info("No existing graph found. Initializing new graph.")

    def shutdown(self, signum=None, frame=None) -> None:
        """Handle graceful shutdown."""
        if signum:
            logging.warning(f"Shutdown signal received ({signum}). Saving state and exiting gracefully.")
        else:
            logging.info("Shutdown requested. Saving state.")
            
        self.shutdown_event.set()
        self.save()
        logging.info("Shutdown complete.")
        
        if signum:  # Only exit if called as signal handler
            exit(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Intelligent semantic filesystem')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a directory')
    index_parser.add_argument('directory', help='Directory to index')
    index_parser.add_argument('--full', action='store_true', 
                            help='Perform full reindexing instead of incremental')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for files')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum results to return')
    search_parser.add_argument('--category', help='Filter by category')
    
    # Regenerate command
    subparsers.add_parser('regenerate', help='Regenerate semantic filesystem')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Remove missing files from index')
    
    # Info command
    subparsers.add_parser('info', help='Show information about the index')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    fs = IntelligentFileSystem()
    
    try:
        if args.command == 'index':
            fs.index_directory(args.directory, incremental=not args.full)
            
        elif args.command == 'search':
            results = fs.semantic_search(args.query, top_k=args.limit, filter_category=args.category)
            print(f"\nSearch results for: '{args.query}'")
            print("-" * 80)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['filepath']} (Score: {result['score']:.4f})")
                metadata = result['metadata']
                if 'category' in metadata:
                    print(f"   Category: {metadata['category']}")
                if 'topic' in metadata:
                    print(f"   Topic: {metadata['topic']}")
                if 'keywords' in metadata:
                    print(f"   Keywords: {metadata['keywords']}")
                print()
                
        elif args.command == 'regenerate':
            fs.regenerate_file_structure()
            
        elif args.command == 'cleanup':
            removed = fs.cleanup()
            print(f"Removed {removed} missing files from index")
            
        elif args.command == 'info':
            doc_count = len(fs.graph.nodes)
            print(f"\nIntelligent File System Info:")
            print("-" * 80)
            print(f"Indexed documents: {doc_count}")
            print(f"Graph database: {Config.GRAPH_DB_PATH}")
            print(f"Semantic filesystem: {Config.LINK_FS_ROOT}")
            
            # Get category statistics
            categories = {}
            for filepath in fs.graph.nodes:
                metadata = fs.graph.nodes[filepath]['metadata']
                category = metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1
                
            print("\nCategories:")
            for category, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"  {category}: {count} files")
                
        else:
            # Default behavior
            print("No command specified. Use --help for usage information.")
    except KeyboardInterrupt:
        fs.shutdown()
    except Exception as e:
        logging.error(f"Error: {e}")
        fs.shutdown()


if __name__ == '__main__':
    main()
