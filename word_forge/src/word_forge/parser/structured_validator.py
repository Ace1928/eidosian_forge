"""
Structured Output Validation for Word Forge.

Provides rigorous schema enforcement and self-healing for LLM-generated lexical data.
Utilizes Pydantic for validation and advanced regex/json-repair for correction.

EIDOSIAN CODE POLISHING PROTOCOL v3.14.15 applied.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, TypeVar

from eidosian_core import eidosian
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Type for generic model binding
M = TypeVar("T", bound=BaseModel)

LOGGER = logging.getLogger("word_forge.parser.validator")


class EnrichmentSchema(BaseModel):
    """
    Strict schema for lexical and semantic enrichment.

    Ensures that LLM outputs follow a predictable structure with
    properly typed and bounded data.
    """

    model_config = {"extra": "ignore"}

    word: str = Field(..., description="The target word being enriched.")
    definition: str = Field("", description="A concise definition of the word.")
    part_of_speech: str = Field("noun", description="The grammatical category.")

    # Semantic Relationships
    synonyms: List[str] = Field(default_factory=list)
    antonyms: List[str] = Field(default_factory=list)
    hypernyms: List[str] = Field(default_factory=list, description="Broader categories.")
    hyponyms: List[str] = Field(default_factory=list, description="Specific types.")
    holonyms: List[str] = Field(default_factory=list, description="The 'whole' this belongs to.")
    meronyms: List[str] = Field(default_factory=list, description="Parts of this word.")

    # Emotional Dimensions
    emotional_valence: float = Field(0.0, ge=-1.0, le=1.0)
    emotional_arousal: float = Field(0.0, ge=0.0, le=1.0)
    connotation: Literal["positive", "negative", "neutral"] = "neutral"

    # Context & Examples
    usage_examples: List[str] = Field(default_factory=list)
    context_domain: str = Field("general", description="The primary domain of use (e.g., technical, medical).")

    @field_validator("word", "definition", "part_of_speech", "context_domain", mode="before")
    @classmethod
    def normalize_strings(cls, v: Any) -> str:
        """Normalize core string fields."""
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("synonyms", "antonyms", "hypernyms", "hyponyms", "holonyms", "meronyms", mode="before")
    @classmethod
    def clean_lists(cls, v: Any) -> List[str]:
        """Clean and normalize string lists."""
        if isinstance(v, str):
            # Split by comma if the LLM provided a string instead of a list
            return [s.strip().lower() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip().lower() for s in v if s]
        return []

    @field_validator("usage_examples", mode="before")
    @classmethod
    def clean_examples(cls, v: Any) -> List[str]:
        """Normalize usage examples into a list."""
        if isinstance(v, str):
            parts = re.split(r"[;\n]+", v)
            return [s.strip() for s in parts if s.strip()]
        if isinstance(v, list):
            return [str(s).strip() for s in v if s]
        return []

    @model_validator(mode="before")
    @classmethod
    def handle_missing_word(cls, data: Any) -> Any:
        """Ensure word key is present if we can infer it."""
        if isinstance(data, dict) and "word" not in data:
            # If the LLM omitted the word but we have it in context,
            # this would be injected by the validator wrapper.
            pass
        return data


class StructuredValidator:
    """
    Self-healing orchestrator for LLM outputs.

    Features:
    - JSON extraction from markdown/text.
    - Syntax repair (basic).
    - Pydantic schema validation.
    - Coercion and default injection.
    """

    def __init__(self, schema: Type[BaseModel] = EnrichmentSchema):
        self.schema = schema
        self.logger = LOGGER

    def _normalize_text(self, text: str) -> str:
        """Normalize quotes and whitespace for more stable parsing."""
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        return text.strip()

    def _extract_brace_blocks(self, text: str) -> List[str]:
        """Extract JSON-like objects using a brace balance scan."""
        blocks: List[str] = []
        depth = 0
        start: Optional[int] = None
        in_string = False
        escaped = False

        for idx, ch in enumerate(text):
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            escaped = False

            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        blocks.append(text[start : idx + 1])
                        start = None
        return blocks

    @eidosian()
    def extract_json_candidates(self, text: str) -> List[str]:
        """Extract possible JSON blocks from text."""
        normalized = self._normalize_text(text)
        candidates: List[str] = []

        for match in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            normalized,
            re.DOTALL | re.IGNORECASE,
        ):
            candidates.append(match.group(1))

        if not candidates:
            candidates.extend(self._extract_brace_blocks(normalized))

        if not candidates:
            brace_match = re.search(r"(\{.*\})", normalized, re.DOTALL)
            if brace_match:
                candidates.append(brace_match.group(1))

        # Preserve order but remove duplicates
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    def _strip_json_comments(self, text: str) -> str:
        """Remove JSON-style comments."""
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    def _quote_unquoted_keys(self, text: str) -> str:
        """Quote unquoted object keys."""
        key_pattern = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)")
        return key_pattern.sub(r'\1"\2"\3', text)

    def _convert_single_quotes(self, text: str) -> str:
        """Convert single-quoted strings to double-quoted strings."""
        return re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'",
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
            text,
        )

    def repair_json(self, json_str: str) -> List[str]:
        """Apply heuristics to fix common LLM JSON errors."""
        base = self._normalize_text(json_str)
        if "{" in base and "}" in base:
            base = base[base.find("{") : base.rfind("}") + 1]

        variants: List[str] = [base]

        cleaned = self._strip_json_comments(base)
        cleaned = cleaned.replace("\t", " ")
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = re.sub(r"\bNone\b", "null", cleaned)
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = self._quote_unquoted_keys(cleaned)
        cleaned = self._convert_single_quotes(cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        variants.append(cleaned)

        compact = re.sub(r"\s+", " ", cleaned).strip()
        variants.append(compact)

        # Preserve order but remove duplicates
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        return unique_variants

    def _coerce_list(self, value: Any, lower: bool = True) -> List[str]:
        if isinstance(value, str):
            parts = re.split(r"[,\n;|]+", value)
            items = [s.strip() for s in parts if s.strip()]
        elif isinstance(value, list):
            items = [str(s).strip() for s in value if s]
        else:
            items = []
        return [item.lower() for item in items] if lower else items

    def _coerce_float(self, value: Any, default: float, min_val: float, max_val: float) -> float:
        try:
            number = float(value)
        except Exception:
            number = default
        return max(min_val, min(max_val, number))

    def _sanitize_data(self, data: Any, context_word: Optional[str]) -> Optional[Dict[str, Any]]:
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                data = data[0]
            else:
                return None
        if not isinstance(data, dict):
            return None

        sanitized: Dict[str, Any] = dict(data)

        if context_word and "word" not in sanitized:
            sanitized["word"] = context_word

        for key in ("definition", "part_of_speech", "context_domain"):
            if key in sanitized and sanitized[key] is not None:
                sanitized[key] = str(sanitized[key]).strip()

        for key in ("synonyms", "antonyms", "hypernyms", "hyponyms", "holonyms", "meronyms"):
            if key in sanitized:
                sanitized[key] = self._coerce_list(sanitized.get(key))

        if "usage_examples" in sanitized:
            sanitized["usage_examples"] = self._coerce_list(sanitized.get("usage_examples"), lower=False)

        if "emotional_valence" in sanitized:
            sanitized["emotional_valence"] = self._coerce_float(sanitized.get("emotional_valence"), 0.0, -1.0, 1.0)
        if "emotional_arousal" in sanitized:
            sanitized["emotional_arousal"] = self._coerce_float(sanitized.get("emotional_arousal"), 0.0, 0.0, 1.0)

        if "connotation" in sanitized:
            connotation = str(sanitized.get("connotation", "")).strip().lower()
            if connotation not in ("positive", "negative", "neutral"):
                if isinstance(sanitized.get("emotional_valence"), (int, float)):
                    valence = float(sanitized["emotional_valence"])
                    if valence > 0.25:
                        connotation = "positive"
                    elif valence < -0.25:
                        connotation = "negative"
                    else:
                        connotation = "neutral"
                else:
                    connotation = "neutral"
            sanitized["connotation"] = connotation

        return sanitized

    def _apply_error_fixes(self, data: Dict[str, Any], errors: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt targeted fixes based on validation errors."""
        fixed = dict(data)
        for err in errors:
            loc = err.get("loc") or ()
            if not loc:
                continue
            field = loc[0]
            if field in (
                "synonyms",
                "antonyms",
                "hypernyms",
                "hyponyms",
                "holonyms",
                "meronyms",
            ):
                fixed[field] = self._coerce_list(fixed.get(field))
            elif field == "usage_examples":
                fixed[field] = self._coerce_list(fixed.get(field), lower=False)
            elif field == "emotional_valence":
                fixed[field] = self._coerce_float(fixed.get(field), 0.0, -1.0, 1.0)
            elif field == "emotional_arousal":
                fixed[field] = self._coerce_float(fixed.get(field), 0.0, 0.0, 1.0)
            elif field == "connotation":
                fixed[field] = "neutral"
            elif field in ("word", "definition", "part_of_speech", "context_domain"):
                fixed[field] = str(fixed.get(field, "")).strip()
        return fixed

    @eidosian()
    def validate(self, raw_text: str, context_word: Optional[str] = None) -> Result[Dict[str, Any]]:
        """
        Parse and validate raw text against the schema.

        Args:
            raw_text: The string output from the LLM.
            context_word: Optional word to inject if missing from output.

        Returns:
            Result object containing the validated dictionary or error details.
        """
        from word_forge.queue.queue_manager import Result  # Import here to avoid circularity if needed

        candidates = self.extract_json_candidates(raw_text)
        if not candidates:
            return Result.failure(
                "NO_JSON_FOUND",
                "Could not find a JSON block in the LLM response.",
            )

        last_error = "Unknown parsing failure"
        for candidate in candidates:
            for repaired in self.repair_json(candidate):
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as e:
                    last_error = f"JSON decoding failed: {str(e)}"
                    continue

                sanitized = self._sanitize_data(data, context_word)
                if sanitized is None:
                    last_error = "Parsed JSON was not an object."
                    continue

                try:
                    validated_model = self.schema.model_validate(sanitized)
                    return Result.success(validated_model.model_dump())
                except ValidationError as e:
                    self.logger.warning(f"Validation failed: {e}")
                    fixed = self._apply_error_fixes(sanitized, e.errors())
                    try:
                        validated_model = self.schema.model_validate(fixed)
                        return Result.success(validated_model.model_dump())
                    except ValidationError as e2:
                        last_error = str(e2)
                        continue

        return Result.failure(
            "SCHEMA_VIOLATION",
            f"Data does not match expected schema: {last_error}",
        )


@eidosian()
def validated_query(
    model_state: Any,
    prompt: str,
    context_word: str,
    max_retries: int = 3,
    validator: Optional[StructuredValidator] = None,
    backoff_base: float = 0.6,
    backoff_jitter: float = 0.2,
) -> Result[Dict[str, Any]]:
    """
    Executes an LLM query with integrated validation and retry logic.

    Self-healing cycle:
    1. Query LLM.
    2. Extract & Validate JSON.
    3. If fail, retry with error feedback in prompt.
    """
    from word_forge.queue.queue_manager import Result

    if validator is None:
        validator = StructuredValidator()

    current_prompt = prompt
    last_error = ""

    for attempt in range(max_retries):
        if attempt > 0:
            LOGGER.info(f"Retry attempt {attempt+1}/{max_retries} for '{context_word}'")
            # Append feedback to prompt
            feedback_prompt = (
                "\n\nYour previous response was invalid.\n"
                f"Error: {last_error}\n"
                "Return ONLY valid JSON matching the schema. "
                "Do not include markdown, commentary, or trailing text."
            )
            current_prompt += feedback_prompt

        temperature = 0.2 if attempt == 0 else 0.0
        max_tokens = 512
        with contextlib.suppress(Exception):
            if hasattr(model_state, "get_model_name") and str(model_state.get_model_name()).startswith("ollama:"):
                max_tokens = 64

        response = model_state.generate_text(
            current_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_beams=1,
        )
        if not response:
            last_error = "Model returned empty response."
            sleep_for = backoff_base * (2**attempt) + random.uniform(0, backoff_jitter)
            time.sleep(sleep_for)
            continue

        result = validator.validate(response, context_word=context_word)
        if result.is_success:
            return result

        last_error = result.error.message if result.error else "Unknown validation error"
        sleep_for = backoff_base * (2**attempt) + random.uniform(0, backoff_jitter)
        time.sleep(sleep_for)

    return Result.failure(
        "MAX_RETRIES_EXCEEDED", f"Failed to get valid output for '{context_word}' after {max_retries} attempts."
    )
