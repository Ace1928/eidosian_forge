#!/usr/bin/env python3
"""
Interactively list, filter, and sort models/datasets/spaces on the Hugging Face Hub,
with local caching of results to avoid unnecessary repeated fetching.

Features:
  - Interactive CLI prompting to choose resource type (model, dataset, or space)
  - Prompt-based filter/sort parameters (task, library, author, search, etc.)
  - Handles pagination (limit), sorting, search, direction, etc.
  - Local JSON cache of previously fetched results. 
    • If the same query is repeated, only new items are added to the cache.
  - Graceful handling of KeyboardInterrupt (Ctrl+C).
  - Modular design for easy extension, maintenance, and future feature addition.

Requires:
  - huggingface_hub >= 0.15.1
  - Python >= 3.7
  - A local environment that can install/execute huggingface_hub

Usage:
  python hf_search_cli.py
  (Then follow the prompts!)

Author:
  Your Name / Organization
"""

import os
import json
import signal
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from huggingface_hub import HfApi
import pickle
import hashlib

# Configurable constants
CACHE_FILE = "hf_hub_cache.pkl"
MAX_RESULTS_DISPLAY = 20
MAX_LIMIT = 10_000  # Prevent excessive API requests


def load_cache(
    cache_path: str = CACHE_FILE,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load cached results using pickle with robust error handling."""
    default_cache = {"models": {}, "datasets": {}, "spaces": {}}

    if not os.path.exists(cache_path):
        return default_cache

    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, PermissionError) as e:
        print(f"⚠️ Cache reset due to error: {e}")
        return default_cache


def save_cache(
    cache: Dict[str, Dict[str, List[Dict[str, Any]]]], cache_path: str = CACHE_FILE
) -> None:
    """Save cache using pickle with atomic write."""
    try:
        directory = os.path.dirname(cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f"{cache_path}.tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_path, cache_path)
    except (IOError, PermissionError) as e:
        print(f"🚨 Failed to save cache: {e}")


def hf_serializer(obj: Any) -> Union[dict, list, str, int, float, bool, None]:
    """Recursively serialize Hugging Face objects with type preservation."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, list):
        return [hf_serializer(item) for item in obj]

    if isinstance(obj, dict):
        return {k: hf_serializer(v) for k, v in obj.items()}

    # Handle Hugging Face objects with __dict__ attribute
    if hasattr(obj, "__dict__"):
        serialized = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
                # Skip methods and non-serializable attributes
                if callable(value):
                    continue
                serialized[key] = hf_serializer(value)
            except Exception as e:
                serialized[key] = f"<SerializationError: {str(e)}>"
        return serialized

    # Fallback for unexpected types
    try:
        return str(obj)
    except Exception as e:
        return f"<Unserializable: {type(obj).__name__}>"


def build_query_key(**kwargs) -> str:
    """Create stable hash key for query parameters with full serialization."""
    cleaned = {k: hf_serializer(v) for k, v in sorted(kwargs.items()) if v is not None}
    return hashlib.sha256(json.dumps(cleaned, sort_keys=True).encode()).hexdigest()


def safe_int_input(prompt: str, default: Optional[int] = None) -> Optional[int]:
    """Validate integer input with error handling."""
    try:
        response = input(prompt).strip()
        return int(response) if response else default
    except ValueError:
        print("⚠️ Invalid number, using default")
        return default


def fetch_models(
    hf_api: HfApi,
    author: Optional[str] = None,
    search: Optional[str] = None,
    task: Optional[str] = None,
    library: Optional[str] = None,
    trained_dataset: Optional[str] = None,
    sort: Optional[str] = None,
    direction: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch models from Hugging Face Hub using huggingface_hub.HfApi with the given filters.
    """
    kwargs = {
        "author": author,
        "search": search,
        "task": task,
        "library": library,
        "trained_dataset": trained_dataset,
        "sort": sort,
        "direction": direction,
        "limit": min(limit or 100, MAX_LIMIT),
        "full": True,
        "token": None,
    }
    models = hf_api.list_models(**kwargs)
    return [hf_serializer(m) for m in models]


def fetch_datasets(
    hf_api: HfApi,
    author: Optional[str] = None,
    search: Optional[str] = None,
    task: Optional[str] = None,
    sort: Optional[str] = None,
    direction: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch datasets from Hugging Face Hub using huggingface_hub.HfApi with the given filters.
    """
    kwargs = {
        "task_categories": task,
        "author": author,
        "search": search,
        "sort": sort,
        "direction": direction,
        "limit": min(limit or 100, MAX_LIMIT),
        "full": True,
        "token": None,
    }
    datasets = hf_api.list_datasets(**kwargs)
    return [hf_serializer(d) for d in datasets]


def fetch_spaces(
    hf_api: HfApi,
    author: Optional[str] = None,
    search: Optional[str] = None,
    sort: Optional[str] = None,
    direction: Optional[int] = None,
    limit: Optional[int] = None,
    # You can add more filters for Spaces if needed (like "datasets", "models", "linked", etc.)
) -> List[Dict[str, Any]]:
    """
    Fetch spaces from Hugging Face Hub using huggingface_hub.HfApi with the given filters.
    Returns a list of raw dictionaries (converted from huggingface_hub.hf_api.SpaceInfo).
    """
    kwargs = {
        "author": author,
        "search": search,
        "sort": sort,
        "direction": direction,
        "limit": min(limit or 100, MAX_LIMIT),
        "token": None,
    }
    spaces = hf_api.list_spaces(**kwargs)
    return [hf_serializer(s) for s in spaces]


def merge_new_results(
    existing: List[Dict[str, Any]], new_data: List[Dict[str, Any]], unique_id_key: str
) -> List[Dict[str, Any]]:
    """
    Merge new_data into existing results, avoiding duplicates.
    Compares by `unique_id_key` field (e.g. 'modelId', 'id', etc.).
    Returns updated merged list.
    """
    existing_ids = {item.get(unique_id_key) for item in existing}
    merged = existing[:]
    for nd in new_data:
        if nd.get(unique_id_key) not in existing_ids:
            merged.append(nd)
    return merged


def display_results(results: List[Dict], resource_type: str) -> None:
    """Uniform results display with pagination."""
    print(f"\n📊 Total results: {len(results)}")
    for idx, item in enumerate(results[:MAX_RESULTS_DISPLAY], 1):
        identifier = item.get("modelId" if resource_type == "models" else "id", "N/A")
        modified = item.get("lastModified", "Unknown")
        print(f"{idx:>3}. {identifier:<60} {modified}")


def handle_api_errors(func):
    """Decorator for API error handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"🚨 API Error: {str(e)}")
            return []

    return wrapper


# Apply error handling to fetch functions
fetch_models = handle_api_errors(fetch_models)
fetch_datasets = handle_api_errors(fetch_datasets)
fetch_spaces = handle_api_errors(fetch_spaces)


def interactive_loop() -> None:
    """
    Main interactive loop. Prompts the user for resource type and relevant filters,
    fetches results from Hugging Face Hub, merges them into local cache, and prints them out.
    """
    hf_api = HfApi()
    cache = load_cache()

    # Trap Ctrl+C to handle partial results gracefully
    def handle_sigint(signum, frame):
        print("\n🛑 Saving cache and exiting.")
        save_cache(cache)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        print("\n=== Hugging Face Hub Explorer ===")
        print("1) Models")
        print("2) Datasets")
        print("3) Spaces")
        print("4) Quit")

        choice = input("Select an option: ").strip()

        if choice == "4":
            save_cache(cache)
            print("\n👋 Goodbye!")
            break

        elif choice not in ("1", "2", "3"):
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            continue

        # Common parameters
        author = input("👤 Filter by author (blank for any): ").strip() or None
        search = input("🔍 Search term (blank for any): ").strip() or None
        sort = input("🔢 Sort field (e.g. downloads, lastModified): ").strip() or None
        direction_str = input("🧭 Sort direction (1=asc, -1=desc): ").strip()
        direction = None
        if direction_str:
            try:
                direction = int(direction_str)
            except ValueError:
                print("⚠️ Invalid input for sort direction, ignoring.")
                direction = None

        limit_str = input("📦 Max results (1-" + str(MAX_LIMIT) + "): ").strip() or None
        limit = None
        if limit_str:
            try:
                limit = int(limit_str)
            except ValueError:
                print("⚠️ Invalid limit value, ignoring.")
                limit = None

        # Build specialized parameters for each resource type
        if choice == "1":
            # Models
            task = input("⚙️ Task (e.g. text-classification): ").strip() or None
            library = input("📚 Library (e.g. pytorch): ").strip() or None
            trained_dataset = input("🎓 Trained dataset: ").strip() or None

            query_key = build_query_key(
                author=author,
                search=search,
                task=task,
                library=library,
                trained_dataset=trained_dataset,
                sort=sort,
                direction=direction,
                limit=limit,
            )

            # Check cache
            cached_results = cache["models"].get(query_key, [])
            print(f"\n📦 Found {len(cached_results)} cached results for this query.")
            if cached_results:
                use_cache = input("♻️ Use cached results only? (y/n): ").strip().lower()
            else:
                use_cache = "n"

            if use_cache == "y":
                results = cached_results
            else:
                # Fetch from Hub
                new_results = fetch_models(
                    hf_api=hf_api,
                    author=author,
                    search=search,
                    task=task,
                    library=library,
                    trained_dataset=trained_dataset,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                )
                # Merge new results into cache
                merged = merge_new_results(cached_results, new_results, "modelId")
                cache["models"][query_key] = merged
                save_cache(cache)
                results = merged

            # Display partial info for demonstration
            display_results(results, "models")

        elif choice == "2":
            # Datasets
            task = input("⚙️ Task category (e.g. text-classification): ").strip() or None

            query_key = build_query_key(
                author=author,
                search=search,
                task=task,
                sort=sort,
                direction=direction,
                limit=limit,
            )

            # Check cache
            cached_results = cache["datasets"].get(query_key, [])
            print(f"\n📦 Found {len(cached_results)} cached results for this query.")
            if cached_results:
                use_cache = input("♻️ Use cached results only? (y/n): ").strip().lower()
            else:
                use_cache = "n"

            if use_cache == "y":
                results = cached_results
            else:
                new_results = fetch_datasets(
                    hf_api=hf_api,
                    author=author,
                    search=search,
                    task=task,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                )
                merged = merge_new_results(cached_results, new_results, "id")
                cache["datasets"][query_key] = merged
                save_cache(cache)
                results = merged

            display_results(results, "datasets")

        else:
            # Spaces
            # We won't prompt for advanced filters like "datasets", "models", "linked" for brevity,
            # but you can extend similarly.
            query_key = build_query_key(
                author=author,
                search=search,
                sort=sort,
                direction=direction,
                limit=limit,
            )

            cached_results = cache["spaces"].get(query_key, [])
            print(f"\n📦 Found {len(cached_results)} cached results for this query.")
            if cached_results:
                use_cache = input("♻️ Use cached results only? (y/n): ").strip().lower()
            else:
                use_cache = "n"

            if use_cache == "y":
                results = cached_results
            else:
                new_results = fetch_spaces(
                    hf_api=hf_api,
                    author=author,
                    search=search,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                )
                merged = merge_new_results(cached_results, new_results, "id")
                cache["spaces"][query_key] = merged
                save_cache(cache)
                results = merged

            display_results(results, "spaces")

        input("\n⏎ Continue...")


def main():
    """
    Main entry point for the script.
    Simply calls interactive_loop().
    """
    print("Welcome to the Hugging Face Hub interactive CLI!")
    interactive_loop()


if __name__ == "__main__":
    main()
