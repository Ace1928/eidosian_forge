"""
Eidosian Hugging Face Model Browser
===================================

This script provides a hyper-efficient, extensible, and modular interface for browsing models
on Hugging Face using the Hugging Face API. Designed with Eidosian principles, it ensures
maximum clarity, flexibility, and performance.

Key Features:
- Efficient model search using Hugging Face API.
- Filtering and sorting based on criteria.
- Clear, self-documenting code with strict type safety.
- Modular design for future extensibility.

Requirements:
- Python 3.10 or higher
- `requests` library (`pip install requests`)

Author:
- Eidos, in collaboration with Lloyd Handyside
"""

from typing import Any, Dict, List, Optional

import requests

# Constants
BASE_URL: str = "https://huggingface.co/api/models"


# Exceptions
class HuggingFaceAPIError(Exception):
    """Raised when Hugging Face API returns an error."""

    pass


class ValidationError(Exception):
    """Raised for invalid input parameters."""

    pass


# API Functions
def fetch_models(
    query: Optional[str] = None,
    limit: int = 10,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches models from Hugging Face API with optional search, filtering, and limit.

    Args:
        query (Optional[str]): Search query for model names or descriptions.
        limit (int): Maximum number of models to fetch. Default is 10.
        filters (Optional[Dict[str, str]]): Additional query filters (e.g., {"pipeline_tag": "text-generation"}).

    Returns:
        List[Dict[str, Any]]: List of model data as dictionaries.

    Raises:
        HuggingFaceAPIError: If the API request fails.
        ValidationError: If the input parameters are invalid.
    """
    if not (1 <= limit <= 100):
        raise ValidationError("Limit must be between 1 and 100.")

    params: Dict[str, str] = {"limit": str(limit)}
    if query:
        params["search"] = query
    if filters:
        params.update(filters)

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise HuggingFaceAPIError(
            f"API Error: {response.status_code} - {response.text}"
        )

    return response.json()


# Filtering and Sorting Functions
def filter_models(
    models: List[Dict[str, Any]], key: str, value: Any
) -> List[Dict[str, Any]]:
    """
    Filters a list of models based on a specific key-value pair.

    Args:
        models (List[Dict[str, Any]]): List of model dictionaries.
        key (str): Model dictionary key to filter by.
        value (Any): Desired value to match.

    Returns:
        List[Dict[str, Any]]: Filtered list of models.
    """
    return [model for model in models if model.get(key) == value]


def sort_models(
    models: List[Dict[str, Any]], key: str, reverse: bool = False
) -> List[Dict[str, Any]]:
    """
    Sorts a list of models based on a specific key.

    Args:
        models (List[Dict[str, Any]]): List of model dictionaries.
        key (str): Model dictionary key to sort by.
        reverse (bool): Sort in descending order if True.

    Returns:
        List[Dict[str, Any]]: Sorted list of models.
    """
    return sorted(models, key=lambda model: model.get(key, 0), reverse=reverse)


# Display Function
def display_models(models: List[Dict[str, Any]]) -> None:
    """
    Displays model information in a readable format.

    Args:
        models (List[Dict[str, Any]]): List of model dictionaries to display.
    """
    if not models:
        print("No models found.")
        return

    print(f"\nFound {len(models)} model(s):")
    for model in models:
        print(
            f"📌 Model: {model.get('id', 'Unknown')} | Downloads: {model.get('downloads', 'N/A')} | Created: {model.get('created_at', 'N/A')}"
        )


# Main Execution
def main() -> None:
    """
    Main entry point for executing the Hugging Face model browser.
    """
    print("Welcome to the Eidosian Hugging Face Model Browser")
    try:
        query = (
            input("Enter search query (leave blank for all models): ").strip() or None
        )
        limit = int(input("Enter the number of models to display (1-100): ").strip())
        pipeline_tag = (
            input(
                "Enter pipeline tag filter (e.g., 'text-generation', leave blank to skip): "
            ).strip()
            or None
        )

        filters = {"pipeline_tag": pipeline_tag} if pipeline_tag else None
        models = fetch_models(query=query, limit=limit, filters=filters)

        # Optional sorting by downloads
        sort_choice = input("Sort by downloads? (y/n): ").strip().lower()
        if sort_choice == "y":
            models = sort_models(models, key="downloads", reverse=True)

        display_models(models)

    except (HuggingFaceAPIError, ValidationError, ValueError) as e:
        print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nProcess interrupted. Goodbye.")


if __name__ == "__main__":
    main()
