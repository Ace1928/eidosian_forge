import re
from typing import List, Dict, Union, Tuple, Any, cast
import logging
import json
import os
import hashlib  # For deterministic LLM suggestion fallback
import requests  # For making HTTP requests to Ollama API
from typing import Callable
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
from flashtext import KeywordProcessor  # type: ignore
import colorspacious  # type: ignore
import shutil

# Configure logging - consider moving this to a central configuration if needed for broader application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_COLOR_MAP_FILE = "trait_color_map.json"  # Default filename for color map
DEFAULT_OLLAMA_API_URL = (
    "http://192.168.4.73:11434/api/generate"  # Default Ollama API endpoint
)


class ColorMapper:
    """
    A class to manage color coding of text segments based on personality traits,
    utilizing a color map and optionally an LLM for suggesting colors and categories.

    The color map JSON structure will now support:
    {
        "version": 1.2,
        "personalities": {
            "neuroticism": {
                "color": [255,0,0],
                "category": "big5",
                "description": "Tendency to experience negative emotions",
                "synonyms": ["anxious", "stress-prone"]
            }
        },
        "traits": {
            "curious": {
                "color": [30, 150, 200],
                "category": "positive",
                "linked_personalities": ["openness"],
                "emotional_correlation": ["interest"]
            }
        },
        "emotions": {
            "joy": {
                "color": [255,215,0],
                "intensity_levels": {
                    "mild": [255,240,100],
                    "strong": [255,200,0]
                }
            }
        },
        "color_rules": {
            "complementary_pairs": [],
            "conflict_indicators": []
        }
    }
    """

    def __init__(
        self,
        max_width: int,
        font: Any,
        color_map_file: str = DEFAULT_COLOR_MAP_FILE,
        ollama_api_url: str = DEFAULT_OLLAMA_API_URL,
    ):
        """
        Initializes the ColorMapper with display parameters, font, color map file path, and Ollama API URL.

        Args:
            max_width (int): Maximum width for text wrapping.
            font (Any): Font object for text rendering.
            color_map_file (str, optional): Path to the JSON file for storing the trait color map.
                Defaults to 'trait_color_map.json'.
            ollama_api_url (str, optional): URL of the Ollama API endpoint for trait suggestions.
                Defaults to 'http://localhost:11434/api/generate'.
        """
        self.max_width = max_width
        self.font = font
        self.space_width = self.font.size(" ")[0]
        self.current_lines: List[List[Dict[str, Union[str, Tuple[int, int, int]]]]] = []
        self.scroll_offset: int = 0
        self.line_height: int = font.get_linesize()
        self.trait_color_map: Dict[str, Dict[str, Union[Tuple[int, int, int], str]]] = (
            {}
        )  # Enhanced color map structure
        self.color_map_file = color_map_file  # Make color_map_file configurable
        self.ollama_api_url = ollama_api_url  # Make Ollama API URL configurable
        self.trait_patterns_cache: Dict[str, re.Pattern] = {}
        self.current_traits_hash: str = ""
        self.personality_map: Dict[str, Dict] = {}
        self.emotion_map: Dict[str, Dict] = {}
        self.color_rules: Dict[str, Any] = {}
        self.llm_usage_metrics = {
            "total_requests": 0,
            "successful_additions": 0,
            "failed_attempts": 0,
            "last_updated": datetime.now().isoformat(),
        }
        self.concept_cache: set[str] = set()  # For quick lookups
        self._load_color_map()  # Load color map on initialization
        self._build_concept_cache()

    def _build_concept_cache(self):
        """Build unified cache with TTL"""
        if getattr(self, "_cache_expiry", 0) < time.time():
            self.concept_cache = (
                set(self.trait_color_map.keys())
                | set(self.personality_map.keys())
                | set(self.emotion_map.keys())
            )
            self._cache_expiry = time.time() + 300  # 5 minute cache

    def _concept_needs_creation(self, concept: str) -> bool:
        """Check if concept exists in any registry"""
        return concept.lower() not in self.concept_cache

    def _load_color_map(self) -> None:
        """
        Loads the trait color map from a JSON file.
        If the file doesn't exist or is invalid, it initializes an empty map and logs a warning/error.
        """
        try:
            if os.path.exists(self.color_map_file):
                with open(self.color_map_file, "r") as f:
                    loaded_data = json.load(f)

                    # Handle legacy format migration
                    if "traits" not in loaded_data:  # Legacy format detection
                        self._migrate_legacy_format(loaded_data)
                    else:
                        self.trait_color_map = loaded_data.get("traits", {})
                        self.personality_map = loaded_data.get("personalities", {})
                        self.emotion_map = loaded_data.get("emotions", {})
                        self.color_rules = loaded_data.get("color_rules", {})
                        logging.info(
                            f"Trait color map loaded from {self.color_map_file} successfully."
                        )
            else:
                self.trait_color_map = {}
                logging.info(
                    f"Trait color map file {self.color_map_file} not found. Initializing with an empty trait color map."
                )
        except json.JSONDecodeError as e:
            self.trait_color_map = {}
            logging.error(
                f"JSONDecodeError loading {self.color_map_file}: {e}. Initializing with an empty trait color map."
            )
        except FileNotFoundError:  # Explicitly catch FileNotFoundError for clarity
            self.trait_color_map = {}
            logging.info(
                f"Trait color map file {self.color_map_file} not found. Initializing with an empty trait color map."
            )
        except Exception as e:
            self.trait_color_map = {}
            logging.error(
                f"Error loading trait color map from {self.color_map_file}: {e}. Initializing with an empty trait color map."
            )

    def _save_color_map(self) -> None:
        """
        Saves the trait color map to a JSON file for persistence.
        Logs success or error messages.
        """
        try:
            # Create timestamped backup
            backup_file = f"{self.color_map_file}.bak.{int(time.time())}"
            if os.path.exists(self.color_map_file):
                shutil.copyfile(self.color_map_file, backup_file)

            full_map = {
                "version": 1.2,
                "traits": self.trait_color_map,
                "personalities": self.personality_map,
                "emotions": self.emotion_map,
                "color_rules": self.color_rules,
            }
            with open(self.color_map_file, "w") as f:
                json.dump(full_map, f, indent=4)
            logging.info(
                f"Trait color map saved to {self.color_map_file} successfully."
            )
        except Exception as e:
            logging.error(f"Error saving trait color map to {self.color_map_file}: {e}")

    def _migrate_legacy_format(self, legacy_data: Dict) -> None:
        """Convert old format to new structure"""
        self.trait_color_map = legacy_data
        for trait, data in self.trait_color_map.items():
            if "category" not in data:
                data["category"] = "legacy"
        logging.info("Migrated legacy color map to new format")

    def _normalize_trait(self, trait: str) -> str:
        """
        Normalizes a trait string to a standard format: lowercase and stripped whitespace.

        Args:
            trait (str): The trait string to normalize.

        Returns:
            str: The normalized trait string.
        """
        return trait.strip().lower()

    def _validate_trait(self, trait: str) -> bool:
        """
        Validates if a trait string is valid based on defined criteria:
        - Not empty.
        - At least 2 characters long.
        - Contains only letters, spaces, and hyphens.

        Args:
            trait (str): The trait string to validate.

        Returns:
            bool: True if the trait is valid, False otherwise.
        """
        trait = trait.strip()
        if not trait:  # Check for empty string first
            logging.warning("Trait is empty and will be ignored.")
            return False
        if len(trait) < 2:
            logging.warning(
                f"Trait '{trait}' is too short (less than 2 characters) and will be ignored."
            )
            return False
        if not re.match(
            r"^[a-zA-Z\s-]+$", trait
        ):  # More restrictive regex for validation
            logging.warning(
                f"Trait '{trait}' contains invalid characters. Only letters, spaces, and hyphens are allowed."
            )
            return False

        # Add reserved keyword check
        RESERVED_KEYWORDS = {"version", "traits", "personalities", "emotions"}
        if trait.lower() in RESERVED_KEYWORDS:
            logging.warning(f"Trait '{trait}' is a reserved keyword and cannot be used")
            return False

        # Add duplicate check
        if self._normalize_trait(trait) in self.concept_cache:
            logging.warning(f"Trait '{trait}' already exists in concept registry")
            return False

        return True

    def _deterministic_suggest_trait_color_and_category(
        self, trait: str
    ) -> Dict[str, Union[Tuple[int, int, int], str]]:
        """
        Generates a deterministic color and category suggestion for a trait using hashing.
        This serves as a fallback when the LLM API is not used or fails.

        Args:
            trait (str): The trait string to suggest color and category for.

        Returns:
            Dict[str, Union[Tuple[int, int, int], str]]: A dictionary containing 'color' (RGB tuple) and 'category' (string).
        """
        trait_hash = int(hashlib.blake2b(trait.encode()).hexdigest(), 16)
        r = (
            trait_hash % 200
        ) + 55  # Ensure colors are not too dark or too bright, adjust ranges as needed
        g = ((trait_hash >> 8) % 200) + 55
        b = ((trait_hash >> 16) % 200) + 55
        color = (r, g, b)
        categories = [
            "positive",
            "negative",
            "neutral",
            "descriptive",
            "contextual",
        ]  # Expanded categories
        category = categories[
            trait_hash % len(categories)
        ]  # Deterministic category assignment
        logging.info(
            f"Deterministic suggestion: color {color} and category '{category}' for trait '{trait}'."
        )
        return {"color": color, "category": category}

    def _get_llm_suggestion_from_api(
        self, trait: str
    ) -> Union[Dict[str, Union[Tuple[int, int, int], str]], None]:
        """
        Calls the Ollama API to get color and category suggestions for a given trait.

        Args:
            trait: The trait string to get suggestions for.

        Returns:
            A dictionary containing 'color' (RGB tuple) and 'category' (string)
            from the LLM suggestion, or None if the API call fails or returns invalid data.
        """
        sanitized_trait: str = re.sub(r"[^\w\s-]", "", trait).strip()
        if not sanitized_trait:
            logging.warning("Trait input is empty or invalid after sanitization.")
            return None

        max_retries: int = 3
        for attempt in range(max_retries):
            try:
                logging.info(
                    f"Attempt {attempt + 1}/{max_retries}: Requesting LLM suggestion for '{sanitized_trait}'"
                )
                response: requests.Response = requests.post(
                    self.ollama_api_url,
                    json={
                        "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",  # Exact model name
                        "prompt": f"""Suggest RGB color and category for personality trait '{sanitized_trait}'. 
                        Respond ONLY with valid JSON like: {{"color": [r,g,b], "category": "category_name"}}
                        No markdown, no explanations, no thinking tags. Use this exact format:""",
                        "stream": False,
                        "options": {"temperature": 0.7, "top_p": 0.9},
                    },
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                llm_response_json: Dict = response.json()

                # Basic validation of LLM response structure
                if not isinstance(llm_response_json, dict):
                    logging.warning(
                        f"LLM API response is not a dictionary: {llm_response_json}"
                    )
                    continue  # Retry

                if (
                    "color" not in llm_response_json
                    or "category" not in llm_response_json
                ):
                    logging.warning(
                        f"LLM API response missing 'color' or 'category' keys: {llm_response_json}"
                    )
                    continue  # Retry

                color_suggestion = llm_response_json.get("color")
                category_suggestion = llm_response_json.get("category")

                if (
                    not isinstance(color_suggestion, list)
                    or len(color_suggestion) != 3
                    or not all(isinstance(c, int) for c in color_suggestion)
                ):
                    logging.warning(
                        f"Invalid 'color' format in LLM API response: {color_suggestion}. Expected [r, g, b] list of integers."
                    )
                    continue  # Retry

                if not isinstance(category_suggestion, str) or not category_suggestion:
                    logging.warning(
                        f"Invalid or empty 'category' in LLM API response: '{category_suggestion}'. Expected a non-empty string."
                    )
                    continue  # Retry

                logging.info(
                    f"LLM API suggested color {color_suggestion} and category '{category_suggestion}' for trait '{sanitized_trait}'."
                )
                return {
                    "color": tuple(color_suggestion),
                    "category": category_suggestion,
                }  # Convert color to tuple

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time: int = 2**attempt
                    logging.warning(
                        f"Network error during API call (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(
                        f"Max retries reached for API call after network errors: {e}"
                    )
                    return None  # Indicate failure after retries
            except requests.HTTPError as e:
                logging.error(
                    f"HTTP error {e.response.status_code} from Ollama API: {e.response.text}"
                )
                return None
            except json.JSONDecodeError as e:
                response_text = (
                    response.text if "response" in locals() else "No response received"
                )
                logging.error(
                    f"Failed to decode JSON response from Ollama API. Response text: {response_text}. Error: {e}"
                )
                return None
            except Exception as e:
                logging.error(f"Unexpected error during Ollama API call: {e}")
                return None

        return None  # Return None if all retries fail due to validation issues

    def _ensure_color_map_has_trait(self, trait: str) -> None:
        """
        Ensures the trait color map contains the given trait.
        If not, it first attempts to get suggestions from the Ollama API.
        If the API call fails or returns invalid suggestions, it falls back to deterministic suggestions.
        If all suggestions fail, it uses default color and category.
        """
        normalized_trait = self._normalize_trait(trait)

        if normalized_trait not in self.trait_color_map:
            # First check personality associations
            personality_matches = [
                p
                for p, data in self.personality_map.items()
                if normalized_trait in data.get("related_traits", [])
            ]

            if personality_matches:
                # Inherit personality color if exists
                personality_color = self.personality_map[personality_matches[0]][
                    "color"
                ]
                self.trait_color_map[normalized_trait] = {
                    "color": personality_color,
                    "category": "personality_derived",
                    "source": f"personality:{personality_matches[0]}",
                }
                return

            logging.info(
                f"Trait '{normalized_trait}' not found in color map. Requesting color and category suggestion."
            )
            suggestion_attempts = 0
            max_suggestion_attempts = 2  # Limit suggestion attempts to prevent infinite loops in case of issues
            llm_suggestion = None  # Initialize outside the loop

            while suggestion_attempts < max_suggestion_attempts:
                suggestion_attempts += 1
                llm_suggestion = self._get_llm_suggestion_from_api(
                    normalized_trait
                )  # Get LLM suggestion from API
                if (
                    llm_suggestion
                    and "color" in llm_suggestion
                    and "category" in llm_suggestion
                ):
                    suggested_color = llm_suggestion["color"]
                    suggested_category = llm_suggestion["category"]
                    # Further validation of LLM suggestion could be added here if needed.
                    self.trait_color_map[normalized_trait] = {
                        "color": suggested_color,
                        "category": suggested_category,
                        "source": "llm_api",  # Indicate source of color/category is LLM API
                    }
                    logging.info(
                        f"Trait '{normalized_trait}' added to color map with LLM API suggested color {suggested_color} and category '{suggested_category}'."
                    )
                    self._save_color_map()  # Persist the updated color map
                    return  # Successfully added, exit loop
                else:
                    logging.warning(
                        f"LLM API suggestion failed for trait '{normalized_trait}' (attempt {suggestion_attempts}/{max_suggestion_attempts}). Retrying..."
                    )

            # If LLM API suggestion fails after max attempts, fallback to deterministic suggestion
            logging.warning(
                f"Falling back to deterministic color/category suggestion for trait '{normalized_trait}'."
            )
            deterministic_suggestion = (
                self._deterministic_suggest_trait_color_and_category(normalized_trait)
            )
            default_color = deterministic_suggestion["color"]
            default_category = deterministic_suggestion["category"]

            if (
                default_color and default_category
            ):  # Ensure deterministic suggestion is valid
                self.trait_color_map[normalized_trait] = {
                    "color": default_color,
                    "category": default_category,
                    "source": "deterministic_fallback",  # Indicate source is deterministic fallback
                }
                logging.info(
                    f"Trait '{normalized_trait}' added to color map with deterministic fallback color {default_color} and category '{default_category}'."
                )
                self._save_color_map()  # Persist with deterministic fallback
                return
            else:  # If even deterministic fallback fails (should not happen in current implementation, but for robustness)
                default_color = (128, 128, 128)  # Default gray if all suggestions fail
                default_category = "unknown"
                self.trait_color_map[normalized_trait] = {
                    "color": default_color,
                    "category": default_category,
                    "source": "default",  # Indicate source is default
                }
                logging.error(
                    f"All LLM API and deterministic suggestions failed for trait '{normalized_trait}'. Using default color {default_color} and category '{default_category}'."
                )
                self._save_color_map()  # Persist even with default

    def _update_trait_patterns(self, traits: List[str]) -> None:
        """Cache compiled regex patterns for current traits"""
        traits_hash = hashlib.md5(json.dumps(traits).encode()).hexdigest()
        if traits_hash != self.current_traits_hash:
            self.trait_patterns_cache = {
                trait: re.compile(rf"\b{re.escape(trait)}\b", re.IGNORECASE)
                for trait in traits
            }
            self.current_traits_hash = traits_hash

    @staticmethod
    def validate_text_input(func):
        @wraps(func)
        def wrapper(instance, text: str, *args, **kwargs):
            if not isinstance(text, str) or not text.strip():
                logging.error("Invalid text input")
                return []
            return func(instance, text, *args, **kwargs)

        return wrapper

    @validate_text_input
    def _apply_color_coding(
        self, text: str, personalities: Dict[str, Any]
    ) -> List[Dict[str, Union[str, Tuple[int, int, int], bool]]]:
        """
        Applies color coding to text based on personality traits, using the enhanced color map with categories.

        Args:
            text (str): The text to apply color coding to.
            personalities (Dict[str, Any]): A dictionary containing personality traits.
                Expected to have a key like 'traits' which is a list of personality trait strings.

        Returns:
            List[Dict[str, Union[str, Tuple[int, int, int], bool]]]: A list of text segments with color coding applied.
        """
        logging.info(
            "Starting _apply_color_coding for text: %s",
            text[:50] + "..." if len(text) > 50 else text,
        )  # Limit logged text for very long inputs
        colored_segments: List[Dict[str, Union[str, Tuple[int, int, int], bool]]] = []

        if not isinstance(text, str):
            logging.error(
                f"Input text is not a string. Got type: {type(text)}. Returning empty segments."
            )
            return colored_segments

        if (
            not isinstance(personalities, dict)
            or not personalities.get("traits")
            or not isinstance(personalities.get("traits"), list)
        ):  # More robust check for personalities format
            logging.warning(
                f"Personalities data is not in the expected format. Expected a dict with 'traits' key as a list. Got: {personalities}. Proceeding without personality-based coloring."
            )
            traits_list: List[str] = []  # Explicitly type traits_list as List[str]

        else:
            traits_list = personalities["traits"]

        if not traits_list:
            logging.info(
                "No valid traits provided in personalities. Proceeding with default black color."
            )
            tokens = re.findall(
                r"\b[\w-]+\b|[^\w\s]", text
            )  # Tokenize even without traits for default coloring
            for token in tokens:
                colored_segments.append(
                    {
                        "text": token,
                        "color": (0, 0, 0),
                        "space_after": not bool(re.match(r"[^\w\s]", token)),
                    }
                )  # Default black color
            return colored_segments

        valid_traits = [
            trait for trait in traits_list if self._validate_trait(trait)
        ]  # Validate traits and filter out invalid ones
        normalized_traits = [
            self._normalize_trait(trait) for trait in valid_traits
        ]  # Normalize traits for consistent matching
        trait_priorities = {
            trait: idx for idx, trait in enumerate(normalized_traits)
        }  # Assign priority weights based on trait order.
        logging.debug(
            f"Traits for color coding: {normalized_traits} with priorities: {trait_priorities}"
        )

        self._update_trait_patterns(
            normalized_traits
        )  # Instead of rebuilding patterns each time
        trait_patterns = self.trait_patterns_cache

        # New: Concept detection pipeline
        self._concept_detection_pipeline(text)

        # Use efficient string searching instead of regex for simple cases
        keyword_processor = KeywordProcessor(case_sensitive=False)
        for trait in normalized_traits:
            keyword_processor.add_keyword(trait)
        found_keywords = keyword_processor.extract_keywords(text)

        # Replace simple tokenization with:
        tokens = re.findall(r"\b[\w'-]+\b|[\W]", text)  # Handle apostrophes and hyphens
        text_lower = text.lower()

        # Check for multi-word traits first
        for trait in normalized_traits:
            if " " in trait and trait in text_lower:
                start_idx = 0
                while (idx := text_lower.find(trait, start_idx)) != -1:
                    end_idx = idx + len(trait)
                    # Add the multi-word match as a special segment
                    colored_segments.append(
                        {
                            "text": text[idx:end_idx],
                            "color": self._get_trait_color(trait),
                            "space_after": True,
                        }
                    )
                    start_idx = end_idx

        logging.info("Completed _apply_color_coding.")
        return colored_segments

    def _concept_detection_pipeline(self, text: str):
        """Unified concept detection and handling"""
        detected_concepts = self._detect_potential_concepts(text)

        for concept in detected_concepts:
            if self._concept_needs_creation(concept):
                self._handle_new_concept(concept)

    def _detect_potential_concepts(self, text: str) -> List[str]:
        """Multi-stage concept detection"""
        # Basic NLP detection
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        # Filter using existing validation
        return [c for c in candidates if self._validate_concept_candidate(c)]

    def _validate_concept_candidate(self, concept: str) -> bool:
        """Programmatic validation before LLM involvement"""
        return (
            len(concept) > 2
            and not concept.isnumeric()
            and any(c.isalpha() for c in concept)
            and concept.lower() not in self.concept_cache
        )

    def _handle_new_concept(self, concept: str):
        """Orchestrate LLM-assisted concept registration"""
        try:
            # Get LLM classification and suggestions
            llm_response = self._get_llm_concept_suggestion(concept)

            if self._validate_llm_response(llm_response):
                self._register_new_concept(concept, llm_response)
                self._update_related_concepts(concept, llm_response)
                self.llm_usage_metrics["successful_additions"] = (
                    cast(int, self.llm_usage_metrics["successful_additions"]) + 1
                )
                self._schedule_map_save()
            else:
                self.llm_usage_metrics["failed_attempts"] = (
                    cast(int, self.llm_usage_metrics["failed_attempts"]) + 1
                )

        except Exception as e:
            logging.error(f"Concept registration failed for {concept}: {e}")
            self._register_fallback_concept(concept)

    def _get_llm_concept_suggestion(self, concept: str) -> Dict:
        """Get structured LLM suggestion for concept type"""
        # Sanitize concept input
        concept = re.sub(r"[^a-zA-Z\s-]", "", concept).strip()
        if not concept:
            return {}

        prompt = f"""Classify and describe '{concept}':
        - Type (trait/personality/emotion)
        - Base color (RGB)
        - Related concepts
        - Category
        - 3 synonyms
        Respond in JSON format"""

        return self._llm_api_call(prompt)

    def _llm_api_call(self, prompt: str) -> Dict:
        """Placeholder for LLM API call - needs implementation"""
        logging.info(f"Calling LLM API with prompt: {prompt}")
        # Replace with actual API call and response handling
        return {}  # Placeholder response

    def _validate_llm_response(self, response: Dict) -> bool:
        """Programmatic validation of LLM output with type safety"""
        required = {
            "type": ["trait", "personality", "emotion"],
            "color": lambda x: isinstance(x, list)
            and len(x) == 3
            and all(isinstance(c, int) for c in x),
            "category": str,
            "synonyms": list,
        }

        try:
            return all(
                (
                    # Handle list type checks
                    (response.get(k) in v)
                    if isinstance(v, list)
                    else
                    # Handle function validators
                    (
                        v(response.get(k))
                        if callable(v)
                        else
                        # Handle type comparisons
                        isinstance(response.get(k), type(v))
                    )
                )
                for k, v in required.items()
            )
        except Exception:
            return False

    def _register_new_concept(self, concept: str, data: Dict):
        """Add to appropriate registry with validation"""
        concept = concept.lower()
        registry_map = {
            "trait": self.trait_color_map,
            "personality": self.personality_map,
            "emotion": self.emotion_map,
        }[data["type"]]

        entry = {
            "color": tuple(data["color"]),
            "category": data["category"],
            "synonyms": data["synonyms"][:3],  # Limit to 3
            "first_detected": datetime.now().isoformat(),
            "llm_generated": True,
        }

        # Add type-specific validation
        if data["type"] == "emotion":
            entry["intensity_levels"] = {"base": entry["color"]}

        registry_map[concept] = entry
        self.concept_cache.add(concept)
        logging.info(f"Registered new {data['type']}: {concept}")

    def _update_related_concepts(self, concept: str, data: Dict):
        """Programmatically link related concepts - Placeholder"""
        logging.info(f"Updating related concepts for {concept} with data: {data}")
        # Implementation to link related concepts based on data from LLM response
        pass

    def _register_fallback_concept(self, concept: str):
        """Register concept with fallback mechanism - Placeholder"""
        logging.warning(f"Registering fallback concept for {concept}")
        # Implementation for fallback concept registration
        pass

    def _schedule_map_save(self):
        """Schedule saving of the color map - Placeholder"""
        logging.info("Scheduling color map save")
        # Implementation for scheduling map save, e.g., using a timer or queue
        pass

    def _is_readable_color(self, color: Tuple[int, int, int]) -> bool:
        """Check if color meets WCAG contrast guidelines against white background"""
        luminance = (
            0.2126 * (color[0] / 255)
            + 0.7152 * (color[1] / 255)
            + 0.0722 * (color[2] / 255)
        )
        return luminance < 0.4  # Simple luminance threshold for dark-enough colors

    def _adjust_color_for_readability(
        self,
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[int, int, int]:
        """
        Adjusts the given color to ensure it meets WCAG AA contrast ratio guidelines
        (4.5:1) against a specified background color.

        This function uses color science principles and the CIELab color space for
        perceptual color adjustments, aiming to enhance readability for accessibility.

        Args:
            color: The RGB color tuple (integers from 0 to 255) to adjust.
            bg_color: The RGB background color tuple (default is white: (255, 255, 255)).

        Returns:
            A tuple representing the adjusted RGB color that meets the contrast ratio
            requirements, or the original color if adjustment is not needed or possible
            without significant color distortion.
        """

        def _relative_luminance(rgb_color: Tuple[int, int, int]) -> float:
            """Calculates relative luminance of an RGB color according to WCAG."""
            r, g, b = [x / 255.0 for x in rgb_color]
            r = (r / 12.92) if (r <= 0.03928) else pow((r + 0.055) / 1.055, 2.4)
            g = (g / 12.92) if (g <= 0.03928) else pow((g + 0.055) / 1.055, 2.4)
            b = (b / 12.92) if (b <= 0.03928) else pow((b + 0.055) / 1.055, 2.4)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def _contrast_ratio(
            color1: Tuple[int, int, int], color2: Tuple[int, int, int]
        ) -> float:
            """Calculates the contrast ratio between two RGB colors according to WCAG."""
            lum1 = _relative_luminance(color1)
            lum2 = _relative_luminance(color2)
            return (
                (lum1 + 0.05) / (lum2 + 0.05)
                if lum1 > lum2
                else (lum2 + 0.05) / (lum1 + 0.05)
            )

        target_contrast_ratio: float = 4.5
        current_contrast_ratio: float = _contrast_ratio(color, bg_color)

        if current_contrast_ratio >= target_contrast_ratio:
            return color  # Color already has sufficient contrast

        # Convert colors to CIELab for lightness adjustment
        lab_color = colorspacious.cspace_convert(color, "sRGB255", "CIELab")
        bg_lab_color = colorspacious.cspace_convert(bg_color, "sRGB255", "CIELab")

        # Determine if we need to increase or decrease lightness based on background lightness
        if (
            lab_color[0] > bg_lab_color[0]
        ):  # Color is lighter than background, need to darken
            adjust_direction = -1
        else:  # Color is darker than background, need to lighten
            adjust_direction = 1

        adjusted_lab_color = list(lab_color)  # Convert to list for modification
        step_size = 2  # Step size for lightness adjustment, adjust as needed

        for _ in range(
            50
        ):  # Limit iterations to prevent infinite loop in extreme cases
            adjusted_lab_color[0] += step_size * adjust_direction
            adjusted_lab_color[0] = max(
                0, min(100, adjusted_lab_color[0])
            )  # Clamp L* value

            # Convert to explicit 3-int tuple
            adjusted_rgb = colorspacious.cspace_convert(
                adjusted_lab_color, "CIELab", "sRGB255"
            )
            adjusted_rgb_tuple = (
                int(adjusted_rgb[0]),
                int(adjusted_rgb[1]),
                int(adjusted_rgb[2]),
            )
            current_contrast_ratio = _contrast_ratio(adjusted_rgb_tuple, bg_color)

            if current_contrast_ratio >= target_contrast_ratio:
                return adjusted_rgb_tuple

            if (
                adjusted_lab_color[0] <= 0 and adjust_direction == -1
            ):  # Reached min lightness and still not enough contrast
                logging.warning(
                    f"Could not achieve sufficient contrast by darkening color {color}. Returning best effort."
                )
                return adjusted_rgb_tuple

            if (
                adjusted_lab_color[0] >= 100 and adjust_direction == 1
            ):  # Reached max lightness and still not enough contrast
                logging.warning(
                    f"Could not achieve sufficient contrast by lightening color {color}. Returning best effort."
                )
                return adjusted_rgb_tuple

        logging.warning(
            f"Color adjustment iterations exceeded for {color}. Returning best effort."
        )
        final_rgb = colorspacious.cspace_convert(
            adjusted_lab_color, "CIELab", "sRGB255"
        )
        return (
            int(final_rgb[0]),
            int(final_rgb[1]),
            int(final_rgb[2]),
        )  # Explicit 3-tuple

    def batch_update_color_map(self, traits: List[str]) -> None:
        """Process multiple traits in batch with single save operation"""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._ensure_color_map_has_trait, t) for t in traits
            ]
            for future in as_completed(futures):
                future.result()
        self._save_color_map()  # Single save after all updates

    @lru_cache(maxsize=1024)
    def _get_trait_color(self, trait: str) -> Tuple[int, int, int]:
        return tuple(self.trait_color_map[trait]["color"])  # type: ignore

    def add_personality_entry(
        self,
        name: str,
        base_color: Tuple[int, int, int],
        category: str,
        description: str = "",
    ):
        """Add structured personality entry"""
        self.personality_map[name.lower()] = {
            "color": base_color,
            "category": category,
            "description": description,
            "related_traits": [],
            "timestamp": datetime.now().isoformat(),
        }
        self._save_color_map()

    def get_emotion_color(
        self, emotion: str, intensity: str = "base"
    ) -> Tuple[int, int, int]:
        """Get color with intensity consideration"""
        emotion_data = self.emotion_map.get(emotion.lower(), {})
        return tuple(
            emotion_data.get("intensity_levels", {}).get(
                intensity, emotion_data.get("color", (128, 128, 128))
            )
        )

    def find_related_concepts(self, trait: str) -> Dict[str, List]:
        """Find connections across the knowledge graph"""
        return {
            "personalities": [
                p
                for p, data in self.personality_map.items()
                if trait in data.get("related_traits", [])
            ],
            "emotions": [
                e
                for e, data in self.emotion_map.items()
                if trait in data.get("associated_traits", [])
            ],
        }


def main():
    try:
        from PIL import ImageFont
    except ImportError:
        print(
            "PIL (Pillow) is not installed. Please install it to use font functionalities."
        )
        return

    try:
        font = ImageFont.load_default()  # Or specify a font file path
    except Exception as e:
        print(
            f"Error loading default font: {e}. Please ensure PIL is correctly installed and fonts are available."
        )
        return

    color_mapper = ColorMapper(max_width=800, font=font)  # Example max_width

    # Configuration from tika_searxng_scraper.py (adjust as needed)
    SEARXNG_URL = "http://192.168.4.73:8888"  # Ensure these are accessible
    TIKA_URL = "http://192.168.4.73:9998"
    PROCESSED_LOG = "processed_urls.json"
    processed_urls = set()
    import os, json, requests

    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            try:
                processed_urls = set(json.load(f))
            except json.JSONDecodeError:
                processed_urls = set()  # Handle empty or corrupted JSON

    def save_processed_url(url):
        processed_urls.add(url)
        with open(PROCESSED_LOG, "w") as f:
            json.dump(list(processed_urls), f, indent=2)

    def search_searxng(query, num_results=5):  # Reduced num_results for demonstration
        from urllib.parse import urljoin

        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,academic",
            "categories": "science",  # or adjust categories as needed
            "safesearch": 1,
            "lang": "en",
            "count": num_results,
        }
        try:
            response = requests.get(
                urljoin(SEARXNG_URL, "/search"), params=params, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Search error: {e}")
            return {"results": []}  # Return empty results to avoid further errors

    def extract_content_with_tika(url):
        try:
            response = requests.get(
                url, stream=True, timeout=20
            )  # Increased timeout for content extraction
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )
            headers = {"Content-type": content_type, "Accept": "application/json"}
            tika_response = requests.put(
                f"{TIKA_URL}/rmeta", headers=headers, data=response.content, timeout=20
            )
            tika_response.raise_for_status()
            return json.loads(tika_response.text)
        except requests.exceptions.RequestException as e:
            print(f"Tika extraction error for URL {url}: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error from Tika for URL {url}: {e}")
            return []
        except Exception as e:
            print(f"General error during Tika processing for URL {url}: {e}")
            return []

    print("--- Autonomous Concept Expansion ---")

    while True:
        topic = input("Enter a topic to explore (or type 'exit' to quit): ").strip()
        if topic.lower() == "exit":
            break
        if not topic:
            print("Please enter a topic.")
            continue

        print(f"Searching SearxNG for: '{topic}'...")
        search_results = search_searxng(topic)

        if not search_results or not search_results.get("results"):
            print(f"No search results found for '{topic}'.")
            continue

        print(f"Found {len(search_results['results'])} results. Processing URLs...")
        for result in search_results["results"]:
            url = result.get("url")
            if url and url not in processed_urls:
                print(f"\nProcessing URL: {url}")
                try:
                    content_parts = extract_content_with_tika(url)
                    if content_parts:
                        extracted_text_list = [
                            part.get("X-TIKA:content", "")
                            for part in content_parts
                            if isinstance(part, dict) and "X-TIKA:content" in part
                        ]
                        full_text_content = "\n".join(extracted_text_list).strip()

                        if full_text_content:
                            print(
                                f"Extracted text content (snippet): {full_text_content[:200]}..."
                            )
                            color_mapper._concept_detection_pipeline(full_text_content)
                            save_processed_url(url)
                            print(f"Processed and saved concepts from: {url}")
                        else:
                            print(f"No significant text content extracted from: {url}")
                    else:
                        print(f"No content parts returned from Tika for: {url}")

                except Exception as e:
                    print(f"Error processing URL {url}: {e}")

        print("\n--- Topic processing completed. ---")
        print(
            "\nUpdated Trait Color Map (snippet):"
        )  # Displaying snippets to avoid overwhelming output
        for i, (trait, data) in enumerate(color_mapper.trait_color_map.items()):
            if i < 5:  # Show first 5 traits as example
                print(f"  {trait}: {data}")
            if i == 5:
                print("  ...")
                break
        print("\nUpdated Personality Map (snippet):")
        for i, (personality, data) in enumerate(color_mapper.personality_map.items()):
            if i < 5:
                print(f"  {personality}: {data}")
            if i == 5:
                print("  ...")
                break
        print("\nUpdated Emotion Map (snippet):")
        for i, (emotion, data) in enumerate(color_mapper.emotion_map.items()):
            if i < 5:
                print(f"  {emotion}: {data}")
            if i == 5:
                print("  ...")
                break

    print("\n--- Autonomous Concept Expansion Completed ---")


if __name__ == "__main__":
    main()
