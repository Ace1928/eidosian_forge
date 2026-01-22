import json
from typing import Any, Callable, Optional, Union
from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown

    An evaluator that calculates the edit distance between JSON strings.

    This evaluator computes a normalized Damerau-Levenshtein distance between two JSON strings
    after parsing them and converting them to a canonical format (i.e., whitespace and key order are normalized).
    It can be customized with alternative distance and canonicalization functions.

    Args:
        string_distance (Optional[Callable[[str, str], float]]): A callable that computes the distance between two strings.
            If not provided, a Damerau-Levenshtein distance from the `rapidfuzz` package will be used.
        canonicalize (Optional[Callable[[Any], Any]]): A callable that converts a parsed JSON object into its canonical string form.
            If not provided, the default behavior is to serialize the JSON with sorted keys and no extra whitespace.
        **kwargs (Any): Additional keyword arguments.

    Attributes:
        _string_distance (Callable[[str, str], float]): The internal distance computation function.
        _canonicalize (Callable[[Any], Any]): The internal canonicalization function.

    Examples:
        >>> evaluator = JsonEditDistanceEvaluator()
        >>> result = evaluator.evaluate_strings(prediction='{"a": 1, "b": 2}', reference='{"a": 1, "b": 3}')
        >>> assert result["score"] is not None

    Raises:
        ImportError: If `rapidfuzz` is not installed and no alternative `string_distance` function is provided.

    