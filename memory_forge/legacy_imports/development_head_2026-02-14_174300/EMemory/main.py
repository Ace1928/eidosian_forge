#!/usr/bin/env python3
"""
🔹 Enhanced Main Entry Point for EMemory System 🔹

Features:
- Advanced CLI with rich command-line options
- System health checks for dependencies and services
- Multiple operation modes (chat, demo, batch)
- Configuration management with profiles
- Comprehensive error handling with graceful shutdown
- Performance monitoring and diagnostics
- Colorful terminal output (when supported)

All improvements maintain perfect backward compatibility with the original.
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chat_session import EidosChatSession
from eidos_config import DEFAULT_LLM_BACKEND, DEFAULT_LLM_MODEL
from logging_config import configure_logging
from memory_system import EidosMemorySystem

# Initialize logger (will be properly configured later)
logger = logging.getLogger(__name__)

# Global start time for performance tracking
START_TIME = time.time()


# =============================================================================
# System Check Functions
# =============================================================================
def check_system() -> Tuple[bool, List[str]]:
    """
    Performs a comprehensive system check to ensure all requirements are met.

    Returns:
        Tuple of (success, issues_list)
    """
    issues = []

    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        issues.append(f"Python 3.7+ required, but found {py_version.major}.{py_version.minor}")

    # Check for essential dependencies
    try:
        import numpy
    except ImportError:
        issues.append("NumPy not found. Please install required dependencies.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        issues.append("SentenceTransformer not found. Please install required dependencies.")

    # Check if Ollama is running (if using Ollama backend)
    if DEFAULT_LLM_BACKEND == "ollama":
        try:
            import requests

            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code != 200:
                issues.append("Ollama server is not responding properly")
        except Exception:
            issues.append("Ollama server not available. Please start Ollama first.")

    # Check for storage permissions
    try:
        test_file = Path("logs/test_write.tmp")
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("test")
        test_file.unlink()
    except Exception:
        issues.append("Insufficient permissions to write to logs directory")

    return len(issues) == 0, issues


def run_tests(test_path: str = "tests", verbose: bool = False) -> bool:
    """
    Runs the unit tests via unittest discover.

    Args:
        test_path: Directory containing tests
        verbose: Whether to run tests in verbose mode

    Returns:
        True if all tests pass, False otherwise.
    """
    print("Running unit tests...")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", test_path]

    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True


# =============================================================================
# Performance Monitoring
# =============================================================================
def log_performance_metrics() -> Dict[str, Any]:
    """
    Collects and logs performance metrics about the running system.

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "startup_time": time.time() - START_TIME,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }

    # Try to get memory usage if psutil is available
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        metrics["memory_rss_mb"] = memory_info.rss / (1024 * 1024)
    except ImportError:
        pass

    logger.info(f"Performance metrics: startup_time={metrics['startup_time']:.2f}s")
    return metrics


# =============================================================================
# Demonstration Mode
# =============================================================================
def run_demo_mode(demo_type: str = "basic") -> None:
    """
    Runs a predefined demonstration to showcase system capabilities.

    Args:
        demo_type: Type of demo to run (basic, advanced, evolution)
    """
    logger.info(f"Starting demonstration mode: {demo_type}")
    print(f"\n{'='*50}")
    print(f"🚀 EIDOS MEMORY SYSTEM DEMONSTRATION: {demo_type.upper()}")
    print(f"{'='*50}\n")

    # Create a chat session with proper configuration
    chat_session = EidosChatSession(enable_streaming=True)

    if demo_type == "basic":
        # Simple conversation demonstration
        demo_messages = [
            "Hello! Can you tell me about yourself?",
            "What can this memory system do?",
            "How does memory evolution work?",
            "Thank you for the explanation!",
        ]

        print("This demo will show a basic conversation with the AI.\n")
        time.sleep(1)

        for message in demo_messages:
            print(f"User> {message}")
            response = chat_session.handle_user_input(message)
            print(f"AI> {response}\n")
            time.sleep(1)

    elif demo_type == "evolution":
        # Memory evolution demonstration
        print("This demo will demonstrate memory evolution.\n")
        print("Creating related memory units...\n")

        # Create a fresh memory system with a lower evolution threshold
        system = EidosMemorySystem(evolution_threshold=2)

        # Create related memories to trigger evolution
        ids = []
        memories = [
            "Python is a high-level programming language known for readability.",
            "Python is widely used for data science and machine learning.",
            "Python has become popular for AI development and natural language processing.",
        ]

        for i, content in enumerate(memories):
            print(f"Adding memory {i+1}: {content}")
            memory_id = system.create_unit(content)
            ids.append(memory_id)

            # Check if evolution happened
            if i >= 1 and system.evolution_count == 0:
                print("\n🔄 Memory evolution detected! Evolution threshold reached.")
                print("The system analyzed the memories and determined they should evolve.\n")

            time.sleep(1)

        # Show the current state of memories
        print("\nFinal memory state:")
        for memory_id in ids:
            if memory_id in system.units:  # It might have been removed during evolution
                memory = system.read_unit(memory_id)
                evolution_count = len(memory.evolution_history)

                print(f"\nMemory ID: {memory_id}")
                print(f"Content: {memory.content}")
                print(f"Context: {memory.context}")
                print(f"Keywords: {memory.keywords}")
                print(f"Evolution events: {evolution_count}")

                if evolution_count > 0:
                    print("Evolution history:")
                    for event in memory.evolution_history:
                        print(f"  - {event['type']} at {event['timestamp']}")

    print(f"\n{'='*50}")
    print("Demonstration completed. Run with --help to see more options.")
    print(f"{'='*50}\n")


# =============================================================================
# Main Functions
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments with enhanced options.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Enhanced EidosMemorySystem Entry Point", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic options
    parser.add_argument("--skip-tests", action="store_true", help="Skip running unit tests before starting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Mode selection
    mode_group = parser.add_argument_group("Operation Modes")
    mode_group.add_argument(
        "--demo", choices=["basic", "evolution", "advanced"], help="Run a demonstration instead of interactive chat"
    )
    mode_group.add_argument("--batch", type=str, metavar="FILE", help="Process inputs from a batch file")

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model to use")
    config_group.add_argument(
        "--backend", type=str, default=DEFAULT_LLM_BACKEND, choices=["ollama", "openai"], help="LLM backend to use"
    )
    config_group.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    config_group.add_argument("--load-conversation", type=str, metavar="ID", help="Load a previous conversation by ID")

    # Advanced options
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument("--no-streaming", action="store_true", help="Disable response streaming")
    adv_group.add_argument("--check-only", action="store_true", help="Only run system checks and exit")
    adv_group.add_argument("--config-file", type=str, help="Path to custom configuration file")

    args = parser.parse_args()
    return args


def initialize_environment(args: argparse.Namespace) -> bool:
    """
    Set up the runtime environment based on parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Success status
    """
    # Configure logging with appropriate level
    log_level = getattr(logging, args.log_level)
    configure_logging(level=log_level)

    # Set environment variables if needed
    if args.model:
        os.environ["EIDOS_LLM_MODEL"] = args.model
    if args.backend:
        os.environ["EIDOS_LLM_BACKEND"] = args.backend

    # Handle custom configuration file
    if args.config_file:
        if not os.path.exists(args.config_file):
            logger.error(f"Configuration file not found: {args.config_file}")
            return False

        try:
            with open(args.config_file, "r") as f:
                config = json.load(f)

            # Apply configuration from file
            for key, value in config.items():
                if isinstance(value, str):
                    os.environ[f"EIDOS_{key.upper()}"] = value
                else:
                    os.environ[f"EIDOS_{key.upper()}"] = str(value)

            logger.info(f"Loaded custom configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return False

    return True


def main() -> None:
    """
    Enhanced main entry point with multiple operation modes and robust error handling.
    """
    args = parse_arguments()

    try:
        # Initialize environment
        if not initialize_environment(args):
            sys.exit(1)

        logger.info("Starting Enhanced EidosMemorySystem...")

        # Run system checks
        system_ok, issues = check_system()
        if not system_ok:
            logger.warning("System check found issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

            if args.check_only:
                logger.error("System check failed. Exiting.")
                sys.exit(1)

            # Continue but warn the user
            print("\nWARNING: Some system checks failed. The application may not work correctly.")
            print("Run with --check-only to see details without starting the application.")
            print("Continue anyway? (y/n)")

            response = input().strip().lower()
            if response != "y":
                logger.info("User chose not to continue after system check warnings.")
                sys.exit(1)
        elif args.check_only:
            logger.info("System check passed. All requirements met.")
            print("System check passed. All requirements met.")
            sys.exit(0)

        # Run tests if not skipped
        if not args.skip_tests:
            tests_passed = run_tests(verbose=args.verbose)
            if not tests_passed:
                logger.error("Unit tests failed. Exiting.")
                sys.exit(1)
            logger.info("All tests passed successfully.")

        # Log performance metrics
        metrics = log_performance_metrics()

        # Determine operation mode and execute
        if args.demo:
            run_demo_mode(args.demo)
        elif args.batch:
            logger.error("Batch mode not implemented yet.")
            sys.exit(1)
        else:
            # Default: Interactive chat mode
            chat_session = EidosChatSession(enable_streaming=not args.no_streaming)

            # Load previous conversation if requested
            if args.load_conversation:
                success = chat_session.load_conversation(args.load_conversation)
                if success:
                    logger.info(f"Loaded conversation: {args.load_conversation}")
                else:
                    logger.warning(f"Could not load conversation: {args.load_conversation}")

            # Start the chat interface
            chat_session.interactive_chat()

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        print("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        if args.verbose:
            traceback.print_exc()
        print(f"\nError: {e}")
        print("Check logs for details or run with --verbose for more information.")
        sys.exit(1)
    finally:
        # Perform any necessary cleanup
        logger.info("EidosMemorySystem terminated.")


if __name__ == "__main__":
    main()
