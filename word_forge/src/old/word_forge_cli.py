import argparse
import datetime
import logging
import os
import sys
import tempfile
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from word_forge.database.database_manager import DBManager, WordEntryDict
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager
from word_forge.worker.worker_thread import (
    ProcessingResult,
    WordForgeWorker,
    create_progress_bar,
    print_table,
)

# This module provides a command-line interface (CLI) for the WordForge application,

# Type aliases for better readability
StatDict = Dict[str, Union[int, float, str, List[Any], Dict[str, Any], None]]
EventDict = Dict[str, Any]


# Safe type conversion helpers
def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int with proper type handling."""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float, str)):
            return int(value)
        return default
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float with proper type handling."""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float, str)):
            return float(value)
        return default
    except (ValueError, TypeError):
        return default


# Argument completion setup
try:
    import argcomplete

    has_argcompletion = True
except ImportError:
    has_argcompletion = False
    argcomplete = None

# Terminal color support
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    has_color = True
except ImportError:

    class _NoColor:
        def __getattr__(self, _):
            return ""

    Fore = Style = _NoColor()
    has_color = False


class CLIState(Enum):
    """States for the WordForge CLI application."""

    INITIALIZING = auto()
    RUNNING = auto()
    INTERACTIVE = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class WordForgeCLI:
    """
    A fully Eidosian CLI for managing the living lexical database,
    providing subcommands and an optional interactive mode.
    """

    def __init__(
        self,
        db_path: str = "word_forge.sqlite",
        data_dir: str = "data",
        log_level: str = "DEBUG",
    ) -> None:
        """
        Initialize the WordForge CLI.

        Args:
            db_path: Path to SQLite database file
            data_dir: Directory containing lexical resources
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Set up logging
        self.logger = logging.getLogger("WordForge")
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = log_level_map.get(log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize state and tracking variables
        self.db_path = db_path
        self.data_dir = data_dir
        self.state = CLIState.INITIALIZING
        self.worker_started = False
        self.start_time = 0
        self.processed_words: Set[str] = set()
        self.worker: Optional[WordForgeWorker] = None
        self.exit_event = threading.Event()
        self._last_stats_time = 0
        self._stats_interval = 10  # seconds between stats updates

        # Error handling
        self._error_count = 0
        self._error_threshold = 5
        self._last_error_time = 0
        self._error_reset_interval = 60  # seconds

        # Initialize components
        self._init_components(db_path, data_dir)
        self.state = CLIState.STOPPED

    def _init_components(self, db_path: str, data_dir: str) -> None:
        """Initialize the core components of WordForge."""
        self.db_manager = DBManager(db_path)
        self.parser_refiner = ParserRefiner()
        self.queue_manager = QueueManager()
        self.emotion_manager = EmotionManager(self.db_manager)
        self.logger.info(f"Initialized with database: {db_path}")
        self.logger.info(f"Data directory: {data_dir}")

    def start_worker(self) -> None:
        """Start the worker thread if not already running."""
        if self.worker_started:
            self.logger.info(f"{Fore.YELLOW}Worker already running.{Style.RESET_ALL}")
            return

        # Check if we're in error circuit breaker mode
        current_time = time.time()
        if (
            self._error_count >= self._error_threshold
            and current_time - self._last_error_time < self._error_reset_interval
        ):
            self.logger.error(
                f"{Fore.RED}Too many errors occurred. Waiting before retry.{Style.RESET_ALL}"
            )
            return

        self.logger.info(f"{Fore.GREEN}Starting WordForge worker...{Style.RESET_ALL}")

        try:
            # Create and start worker
            self.worker = WordForgeWorker(
                parser_refiner=self.parser_refiner,
                queue_manager=self.queue_manager,
                db_manager=self.db_manager,
                emotion_manager=self.emotion_manager,
                result_callback=self._on_word_processed,
                logger=self.logger,
            )

            self.worker.start()
            self.worker_started = True
            self.start_time = time.time()
            self.state = CLIState.RUNNING
            self.logger.info(
                f"{Fore.GREEN}Worker started successfully.{Style.RESET_ALL}"
            )

            # Reset error count on successful start
            self._error_count = 0

            # Log initial statistics
            current_time = time.time()
            if current_time - self._last_stats_time >= self._stats_interval:
                self._log_stats_summary(self.worker.get_statistics())
                self._last_stats_time = current_time
        except Exception as e:
            self._error_count += 1
            self._last_error_time = time.time()
            self.logger.error(
                f"{Fore.RED}Failed to start worker: {str(e)}{Style.RESET_ALL}"
            )
            self.state = CLIState.ERROR

    def _on_word_processed(self, result: ProcessingResult) -> None:
        """
        Handle word processing results from the worker.

        Args:
            result: Processing result object from worker
        """
        try:
            # Handle different ProcessingResult types (dict-like or object-like)
            if result:
                term = result.get("term", "")
                success = result.get("success", False)
            else:
                # Assume object with attributes
                term = getattr(result, "term", "")
                success = getattr(result, "success", False)

            if success:
                self.processed_words.add(term)

                # Only update stats at appropriate intervals to avoid overhead
                current_time = time.time()
                if current_time - self._last_stats_time >= self._stats_interval:
                    self._last_stats_time = current_time
                    if self.worker is not None:
                        stats = self.worker.get_statistics()
                        self._log_stats_summary(stats)
        except Exception as e:
            self.logger.error(f"Error processing result: {str(e)}")
            # Continue operation despite errors

    def _log_stats_summary(self, stats: StatDict) -> None:
        """Log a summary of current statistics."""
        proc_count = safe_int(stats.get("processed_count", 0))
        proc_rate = safe_float(stats.get("processing_rate_per_minute", 0.0))
        queue_size = safe_int(stats.get("queue_size", 0))
        error_count = safe_int(stats.get("error_count", 0))

        self.logger.info(
            f"{Fore.CYAN}Stats: {proc_count} words processed "
            f"({proc_rate:.2f}/min), {queue_size} in queue, {error_count} errors{Style.RESET_ALL}"
        )

    def stop_worker(self) -> None:
        """Stop the worker thread if running and wait for it to finish."""
        if not self.worker_started or self.worker is None:
            self.logger.info(
                f"{Fore.YELLOW}Worker thread is not running.{Style.RESET_ALL}"
            )
            return

        self.logger.info(f"{Fore.GREEN}Stopping worker thread...{Style.RESET_ALL}")
        self.worker.request_stop()
        self.worker.join(timeout=5.0)
        self.worker_started = False

        # Print final statistics
        if self.start_time > 0:
            runtime = time.time() - self.start_time
            formatted_runtime = str(datetime.timedelta(seconds=int(runtime)))
            stats: StatDict = self.worker.get_statistics()

            self.logger.info(
                f"\n{Fore.CYAN}=== Processing Summary ==={Style.RESET_ALL}"
            )
            self.logger.info(f"Runtime: {formatted_runtime}")
            self.logger.info(
                f"Words processed: {safe_int(stats.get('processed_count', 0))}"
            )

            processing_rate = safe_float(stats.get("processing_rate_per_minute", 0.0))
            self.logger.info(f"Processing rate: {processing_rate:.2f} words/minute")

            self.logger.info(f"Words in queue: {safe_int(stats.get('queue_size', 0))}")
            self.logger.info(
                f"Unique words seen: {safe_int(stats.get('total_unique_words', 0))}"
            )
            self.logger.info(f"Error count: {safe_int(stats.get('error_count', 0))}")

    def run_demo(self, minutes: float = 1.0) -> None:
        """
        Run a demonstration of WordForge for a specified duration.

        Args:
            minutes: Duration to run the demo in minutes
        """
        run_seconds = minutes * 60

        # Create some demo files
        temp_dir = tempfile.mkdtemp()
        demo_data_dir = os.path.join(temp_dir, "data")
        os.makedirs(demo_data_dir, exist_ok=True)

        # Create sample dictionary files
        self._create_demo_files(Path(demo_data_dir))

        # Reinitialize with demo data
        self._init_components(
            os.path.join(temp_dir, "word_forge_demo.sqlite"), demo_data_dir
        )

        # Seed the queue with initial words
        seed_words = [
            "algorithm",
            "language",
            "computer",
            "data",
            "program",
            "network",
            "code",
            "function",
            "logic",
            "system",
        ]

        self.logger.info(f"Seeding queue with {len(seed_words)} initial words...")
        for word in seed_words:
            self.add_word(word)

        # Start worker
        self.logger.info(f"Starting worker for {minutes} minute(s)...")
        self.start_worker()

        # Monitor progress during runtime with periodic status updates
        update_interval = 5  # Show stats every 5 seconds
        start_time = time.time()
        end_time = start_time + run_seconds

        try:
            while time.time() < end_time and self.worker_started:
                # Sleep for the update interval or until end time
                remaining = min(update_interval, end_time - time.time())
                if remaining <= 0:
                    break
                time.sleep(remaining)

                # Show progress
                if self.worker is not None:
                    # Display current statistics
                    stats = self.worker.get_statistics()
                    self._log_stats_summary(stats)

                elapsed = time.time() - start_time
                # Convert to integers to satisfy create_progress_bar's type requirements
                self.logger.info(
                    f"Progress: {create_progress_bar(int(elapsed), int(run_seconds))}"
                )

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Stopping gracefully...")

        finally:
            # Display final stats and summary
            self.stop_worker()
            self.logger.info(f"Demo files at: {temp_dir}")
            self.logger.info("Demo complete!")

    def _create_demo_files(self, data_dir: Path) -> None:
        """
        Create sample dictionary files for the demo.

        Args:
            data_dir: Directory to write demo files to
        """
        # Create a simple demo dictionary with explicit type annotation
        demo_dict: Dict[str, List[Dict[str, Any]]] = {
            "words": [
                {
                    "term": "algorithm",
                    "definition": "A step-by-step procedure for solving a problem",
                    "part_of_speech": "noun",
                    "usage_examples": [
                        "The sorting algorithm arranges elements in order."
                    ],
                },
                {
                    "term": "recursion",
                    "definition": "A method where the solution depends on solutions to smaller instances of the same problem",
                    "part_of_speech": "noun",
                    "usage_examples": [
                        "The function uses recursion to traverse the tree."
                    ],
                },
                {
                    "term": "function",
                    "definition": "A named section of a program that performs a specific task",
                    "part_of_speech": "noun",
                    "usage_examples": [
                        "This function calculates the average of a list of numbers."
                    ],
                },
            ]
        }

        # Write demo dictionary to file
        import json

        with open(data_dir / "demo_dictionary.json", "w") as f:
            json.dump(demo_dict, f, indent=2)

        # Create a sample word list
        with open(data_dir / "sample_words.txt", "w") as f:
            f.write(
                "\n".join(
                    [
                        "programming",
                        "database",
                        "interface",
                        "object",
                        "class",
                        "inheritance",
                        "polymorphism",
                        "variable",
                        "constant",
                        "library",
                    ]
                )
            )

    def list_queue(self) -> None:
        """Display the current queue size and a sample of queued items."""
        queue_size = self.queue_manager.size
        self.logger.info(f"Current queue size: {queue_size}")

        if queue_size > 0:
            # Get up to 5 items from the queue without removing them
            sample_result = self.queue_manager.get_sample(5)
            if hasattr(sample_result, "value") and sample_result.value:
                self.logger.info("Sample queue items:")
                for item in sample_result.value:
                    self.logger.info(f"  - {item}")
            else:
                self.logger.warning(
                    f"Failed to get queue sample: {getattr(sample_result, 'error', 'Unknown error')}"
                )

    def shutdown(self) -> None:
        """Gracefully shut down all components."""
        self.state = CLIState.STOPPING
        if self.worker_started:
            self.stop_worker()
        self.exit_event.set()
        self.state = CLIState.STOPPED
        self.logger.info("WordForge shutdown complete.")

    def _display_help(self) -> None:
        """Display help for interactive mode."""
        print(f"\n{Fore.CYAN}=== WordForge Commands ==={Style.RESET_ALL}")
        commands = [
            ("help", "Show this help message"),
            ("start", "Start the worker"),
            ("stop", "Stop the worker"),
            ("add <word>", "Add a word to the processing queue"),
            ("batch <file>", "Add words from a file (one per line)"),
            ("search <word>", "Search for a word in the database"),
            ("queue", "Show current queue status"),
            ("stats", "Show detailed statistics"),
            ("exit", "Exit WordForge"),
        ]

        for cmd, desc in commands:
            print(f"{Fore.GREEN}{cmd.ljust(15)}{Style.RESET_ALL}{desc}")

    def add_word(self, term: str):
        """
        Enqueue a word for processing.

        Args:
            term: The word to add to the processing queue

        Returns:
            Whether the word was successfully enqueued
        """
        enqueued = self.queue_manager.enqueue(term)
        if enqueued:
            self.logger.info(
                f"{Fore.GREEN}Enqueued '{term}' for processing.{Style.RESET_ALL}"
            )
        else:
            self.logger.info(
                f"{Fore.YELLOW}'{term}' is already queued or processed.{Style.RESET_ALL}"
            )
        return enqueued

    def batch_add_words(self, file_path: str) -> Tuple[int, int]:
        """
        Add multiple words from a file (one word per line).

        Args:
            file_path: Path to file containing words

        Returns:
            Tuple of (enqueued count, total word count)
        """
        try:
            with open(file_path, "r") as f:
                words = [line.strip() for line in f if line.strip()]

            self.logger.info(f"Processing {len(words)} words from {file_path}")

            enqueued_count = 0
            for word in words:
                if self.add_word(word):
                    enqueued_count += 1

            return enqueued_count, len(words)

        except Exception as e:
            self.logger.error(f"Error reading batch file: {str(e)}")
            return 0, 0

    def search_word(self, term: str) -> Optional[WordEntryDict]:
        """
        Retrieve and display a word entry from the database.

        Args:
            term: The word to search for

        Returns:
            The word entry if found, None otherwise
        """
        try:
            entry = self.db_manager.get_word_entry(term.lower())
            if not entry:
                self.logger.info(
                    f"{Fore.RED}No entry found for '{term}'.{Style.RESET_ALL}"
                )
                return None

            # Display the entry in a user-friendly manner
            self.logger.info(f"{Fore.CYAN}=== Entry for '{term}' ==={Style.RESET_ALL}")
            self.logger.info(f"Definition: {entry['definition']}")
            self.logger.info(f"Part of Speech: {entry['part_of_speech']}")
            self.logger.info("Usage Examples:")
            for example in entry["usage_examples"]:
                self.logger.info(f"  - {example}")
            self.logger.info("Relationships:")
            for rel in entry["relationships"]:
                self.logger.info(
                    f"  - {rel['relationship_type']}: {rel['related_term']}"
                )

            return entry

        except Exception as e:
            self.logger.error(f"Error searching for word: {str(e)}")
            return None

    def display_statistics(self) -> None:
        """Display detailed statistics about the worker and database."""
        if not self.worker_started or self.worker is None:
            self.logger.info(
                f"{Fore.YELLOW}Worker not running. No statistics available.{Style.RESET_ALL}"
            )
            return

        stats: StatDict = self.worker.get_statistics()

        # Print worker status
        state_str_value = stats.get("state", "UNKNOWN")
        state_str = str(state_str_value) if state_str_value is not None else "UNKNOWN"

        status_color_map = {
            "RUNNING": Fore.GREEN,
            "PAUSED": Fore.YELLOW,
            "STOPPING": Fore.RED,
            "STOPPED": Fore.RED,
            "ERROR": Fore.RED,
        }
        status_color = status_color_map.get(state_str, Fore.WHITE)

        # Format runtime
        runtime_secs = stats.get("runtime_seconds", 0)
        if isinstance(runtime_secs, (int, float)):
            formatted_runtime = str(datetime.timedelta(seconds=int(runtime_secs)))
        else:
            formatted_runtime = "Unknown"

        # Create and print statistics table
        headers = ["Metric", "Value"]
        rows: List[List[str]] = []

        def format_stat(key: str, default: str, formatter: Optional[str] = None) -> str:
            """Format a statistic with proper type handling"""
            value = stats.get(key, default)
            if formatter and isinstance(value, (int, float)):
                if formatter == "int":
                    return str(int(value))
                elif formatter == "float1":
                    return f"{float(value):.1f}"
                elif formatter == "float2":
                    return f"{float(value):.2f}"
                elif formatter == "float3":
                    return f"{float(value):.3f}"
                elif formatter == "ms":
                    return f"{float(value) * 1000:.2f} ms"
            return str(value)

        rows = [
            ["Status", f"{status_color}{state_str}{Style.RESET_ALL}"],
            ["Runtime", formatted_runtime],
            ["Words Processed", format_stat("processed_count", "0", "int")],
            [
                "Processing Rate",
                format_stat("processing_rate_per_minute", "0.0", "float2")
                + " words/min",
            ],
            ["Avg Processing Time", format_stat("avg_processing_time", "0.0", "ms")],
            ["Queue Size", format_stat("queue_size", "0", "int")],
            ["Total Unique Words", format_stat("total_unique_words", "0", "int")],
            ["Error Count", format_stat("error_count", "0", "int")],
            ["Last Processed", format_stat("last_processed", "None")],
        ]

        self.logger.info(f"\n{Fore.CYAN}=== WordForge Statistics ==={Style.RESET_ALL}")
        print_table(headers, rows)

        # Show progress bar if processing
        if state_str == "RUNNING":
            processed_value = stats.get("processed_count", 0)
            queue_size_value = stats.get("queue_size", 0)

            # Ensure values are integers
            processed = safe_int(processed_value)
            queue_size = safe_int(queue_size_value)

            if processed > 0 and queue_size > 0:
                total = processed + queue_size
                self.logger.info(f"\nProgress: {create_progress_bar(processed, total)}")

        # Show recent activities
        recent_events_value = stats.get("recent_events", [])
        recent_events: List[EventDict] = []

        # Ensure recent_events is a list
        if isinstance(recent_events_value, list):
            recent_events = recent_events_value

        if recent_events:
            self.logger.info(f"\n{Fore.CYAN}Recent Activity:{Style.RESET_ALL}")
            for event in recent_events[-3:]:  # Show last 3 events
                # Safe handling of event data
                event_time_raw = event.get("timestamp")
                event_type = event.get("event_type", "")
                term = str(event.get("term", "unknown"))
                error = str(event.get("error", "unknown error"))

                # Safe timestamp conversion
                if isinstance(event_time_raw, (int, float)):
                    event_time = datetime.datetime.fromtimestamp(
                        event_time_raw
                    ).strftime("%H:%M:%S")
                else:
                    event_time = "??:??:??"

                if event_type == "word_processed":
                    self.logger.info(f"  {event_time} - Processed: {term}")
                elif event_type == "error":
                    self.logger.info(f"  {event_time} - Error: {error}")
                elif event_type == "state_change":
                    details_value = event.get("details", {})
                    if isinstance(details_value, dict):
                        details: Dict[str, Any] = details_value
                        old_state = str(details.get("old_state", "?"))
                        new_state = str(details.get("new_state", "?"))
                        self.logger.info(
                            f"  {event_time} - State changed: {old_state} → {new_state}"
                        )

    def interactive_shell(self) -> None:
        """
        Launch an interactive REPL-like shell for WordForge.

        Commands:
          start      - start the worker
          stop       - stop the worker
          add <word> - enqueue a word
          search <word> - lookup word in DB
          queue      - show queue size
          stats      - show detailed statistics
          exit       - stop worker & quit
        """
        self.state = CLIState.INTERACTIVE
        print(
            f"{Fore.GREEN}Entering WordForge Interactive Mode. Type 'help' for commands.{Style.RESET_ALL}"
        )
        self.start_worker()  # Start worker by default in interactive mode

        while True:
            try:
                cmd = input(f"\n{Fore.MAGENTA}WordForge> {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "exit"

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            main_cmd = parts[0].lower()

            if main_cmd == "help":
                self._display_help()
            elif main_cmd == "start":
                self.start_worker()
            elif main_cmd == "stop":
                self.stop_worker()
            elif main_cmd == "add":
                if len(parts) < 2:
                    print("Usage: add <word>")
                else:
                    word = parts[1]
                    self.add_word(word)
            elif main_cmd == "batch":
                if len(parts) < 2:
                    print("Usage: batch <file_path>")
                else:
                    file_path = parts[1]
                    enqueued, total = self.batch_add_words(file_path)
                    print(f"Enqueued {enqueued} of {total} words from {file_path}")
            elif main_cmd == "search":
                if len(parts) < 2:
                    print("Usage: search <word>")
                else:
                    word = parts[1]
                    self.search_word(word)
            elif main_cmd == "queue":
                self.list_queue()
            elif main_cmd == "stats":
                self.display_statistics()
            elif main_cmd == "exit":
                self.shutdown()
                break
            else:
                print(f"Unknown command: {cmd}")


def main() -> None:
    """
    The main entry point for the WordForge CLI. Provides subcommands
    for an all-in-one lexical database solution.
    """
    parser = argparse.ArgumentParser(
        description="WordForge: A Living, Self-Refining Lexical Database."
    )
    parser.add_argument(
        "--db-path",
        default="word_forge.sqlite",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing lexical resource files (JSON, TTL, etc.).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    subparsers = parser.add_subparsers(
        title="Commands", description="Available operations", dest="command"
    )

    # Subcommand: start (starts the background worker)
    start_parser = subparsers.add_parser(
        "start", help="Start the worker in background."
    )
    start_parser.add_argument(
        "--minutes",
        type=float,
        default=0,
        help="Run for specified minutes then stop (0 for indefinite).",
    )

    # Subcommand: stop (stops the background worker)
    subparsers.add_parser("stop", help="Stop the background worker.")

    # Subcommand: add-word
    add_parser = subparsers.add_parser(
        "add-word", help="Enqueue a word for processing."
    )
    add_parser.add_argument("word", help="The word to enqueue.")

    # Subcommand: batch
    batch_parser = subparsers.add_parser(
        "batch", help="Add multiple words from a file."
    )
    batch_parser.add_argument("file", help="Path to file with words (one per line).")

    # Subcommand: search
    search_parser = subparsers.add_parser("search", help="Search for a word in the DB.")
    search_parser.add_argument("word", help="The word to look up.")

    # Subcommand: queue
    subparsers.add_parser("queue", help="Show the current queue size.")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Show detailed statistics.")

    # Subcommand: interactive
    subparsers.add_parser("interactive", help="Enter interactive mode.")

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo", help="Run a demonstration with sample data."
    )
    demo_parser.add_argument(
        "--minutes", type=float, default=1.0, help="Duration of the demo in minutes."
    )

    if has_argcompletion and argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()
    cli = WordForgeCLI(
        db_path=args.db_path, data_dir=args.data_dir, log_level=args.log_level
    )

    # If no subcommand provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommands
    try:
        if args.command == "start":
            # Add seed words to ensure there's something to process
            seed_words = [
                "algorithm",
                "knowledge",
                "intelligence",
                "system",
                "recursive",
            ]
            for word in seed_words:
                cli.add_word(word)

            cli.start_worker()

            # If time-limited run was requested
            if args.minutes > 0:
                end_time = time.time() + (args.minutes * 60)
                try:
                    while time.time() < end_time and not cli.exit_event.is_set():
                        # Show statistics every 10 seconds
                        cli.display_statistics()
                        time.sleep(10)
                    cli.stop_worker()
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    cli.stop_worker()
            else:
                # Indefinite run with stats display until interrupted
                print("Worker started. Press Ctrl+C to stop.")
                try:
                    while not cli.exit_event.is_set():
                        cli.display_statistics()
                        time.sleep(10)
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    cli.stop_worker()

        elif args.command == "stop":
            cli.stop_worker()

        elif args.command == "add-word":
            cli.start_worker()  # Ensure worker is running
            cli.add_word(args.word)
            if cli.worker_started:
                cli.stop_worker()

        elif args.command == "batch":
            cli.start_worker()  # Ensure worker is running
            enqueued, total = cli.batch_add_words(args.file)
            print(f"Enqueued {enqueued} of {total} words from {args.file}")
            if cli.worker_started:
                cli.stop_worker()

        elif args.command == "search":
            cli.search_word(args.word)

        elif args.command == "queue":
            cli.list_queue()

        elif args.command == "stats":
            worker_started_internally = False
            if not cli.worker_started:
                cli.start_worker()
                worker_started_internally = True

            cli.display_statistics()

            if worker_started_internally:
                cli.stop_worker()

        elif args.command == "interactive":
            cli.interactive_shell()

        elif args.command == "demo":
            cli.run_demo(args.minutes)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if hasattr(cli, "state") and cli.state != CLIState.STOPPED:
            cli.shutdown()


if __name__ == "__main__":
    main()
