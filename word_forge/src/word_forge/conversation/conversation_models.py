from eidosian_core import eidosian

"""
Protocols and model implementations for the multi-model conversation system.
"""

import json
import random  # Import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

# Import Result and Error types correctly
from word_forge.configs.config_essentials import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
)

# Import model protocols and types from conversation_types
from word_forge.conversation.conversation_types import (
    AffectiveLexicalModel,
    IdentityModel,
    LightweightModel,
    MessageDict,  # Import MessageDict
    ModelContext,
    ReflexiveModel,  # Import ReflexiveModel protocol
)
from word_forge.parser.language_model import ModelState

# --- LLM Implementations ---


@dataclass
class ReflexiveLanguageModel(ReflexiveModel):
    """LLM-backed reflexive model for fast context updates."""

    llm_state: ModelState
    max_new_tokens: int = 64
    temperature: float = 0.2

    @eidosian()
    def generate_reflex(self, context: ModelContext) -> Result[ModelContext]:
        """Generate a short reflexive update using the configured LLM."""
        prompt = (
            "Provide a brief reflexive note based on the user input.\n"
            f"Input: {context.get('current_input', '')}\n"
            "Reflexive note:"
        )
        output = self.llm_state.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        if not output:
            error = Error.create(
                message="Reflexive model failed to generate output.",
                code="REFLEXIVE_GENERATION_FAILED",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
            )
            return Result[ModelContext].failure(
                error.code,
                error.message,
                error.context,
                error.category,
                error.severity,
            )

        context["reflexive_output"] = {
            "timestamp": time.time(),
            "note": output.strip(),
        }
        return Result[ModelContext].success(context)


@dataclass
class LightweightLanguageModel(LightweightModel):
    """LLM-backed lightweight model for routing and quick analysis."""

    llm_state: ModelState
    max_new_tokens: int = 64
    temperature: float = 0.3

    @eidosian()
    def process(self, context: ModelContext) -> Result[ModelContext]:
        """Process context and decide a routing hint using the configured LLM."""
        prompt = (
            "Classify the routing need for this user input as one of:\n"
            "standard, escalated, informational.\n"
            f"Input: {context.get('current_input', '')}\n"
            "Routing decision:"
        )
        output = self.llm_state.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        if not output:
            error = Error.create(
                message="Lightweight model failed to generate output.",
                code="LIGHTWEIGHT_GENERATION_FAILED",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
            )
            return Result[ModelContext].failure(
                error.code,
                error.message,
                error.context,
                error.category,
                error.severity,
            )

        decision_text = output.strip().lower()
        if "escalated" in decision_text:
            decision = "escalated"
        elif "informational" in decision_text:
            decision = "informational"
        else:
            decision = "standard"

        if "additional_data" not in context:
            context["additional_data"] = {}
        context["additional_data"]["routing_decision"] = decision
        return Result[ModelContext].success(context)


@dataclass
class AffectiveLexicalLanguageModel(AffectiveLexicalModel):
    """LLM-backed affective model for core response generation."""

    llm_state: ModelState
    max_new_tokens: int = 128
    temperature: float = 0.7

    @eidosian()
    def generate_core_response(self, context: ModelContext) -> Result[ModelContext]:
        """Generate the intermediate response and affective state."""
        prompt = (
            "Generate a concise assistant response to the user input.\n"
            f"Input: {context.get('current_input', '')}\n"
            "Response:"
        )
        output = self.llm_state.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        if not output:
            error = Error.create(
                message="Affective model failed to generate output.",
                code="AFFECTIVE_GENERATION_FAILED",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
            )
            return Result[ModelContext].failure(
                error.code,
                error.message,
                error.context,
                error.category,
                error.severity,
            )

        context["intermediate_response"] = output.strip()

        emotion_manager = context.get("emotion_manager")
        if emotion_manager:
            valence, arousal = emotion_manager.analyze_text_emotion(context.get("current_input", ""))
            context["affective_state"] = {
                "valence": valence,
                "arousal": arousal,
            }

        return Result[ModelContext].success(context)


# --- Mock Implementations ---


@dataclass
class MockReflexiveModel(ReflexiveModel):  # Inherit from protocol
    """Mock implementation for rapid initial response."""

    delay: float = 0.05

    @eidosian()
    def generate_reflex(self, context: ModelContext) -> Result[ModelContext]:
        """Simulates generating a quick reflex or context update."""
        time.sleep(self.delay)
        context["reflexive_output"] = {
            "timestamp": time.time(),
            "note": f"Reflex triggered by input: {context.get('current_input', '')[:20]}...",
        }
        return Result[ModelContext].success(context)


@dataclass
class MockLightweightModel(LightweightModel):  # Inherit from protocol
    """Mock implementation for routing and basic processing."""

    delay: float = 0.1

    @eidosian()
    def process(self, context: ModelContext) -> Result[ModelContext]:
        """Simulates processing context, maybe routing or simple analysis."""
        time.sleep(self.delay)
        if "additional_data" not in context:
            context["additional_data"] = {}  # Ensure key exists
        context["additional_data"]["routing_decision"] = random.choice(["standard", "escalated", "informational"])
        return Result[ModelContext].success(context)


@dataclass
class MockAffectiveLexicalModel(AffectiveLexicalModel):  # Inherit from protocol
    """Mock implementation for core understanding and response generation."""

    delay: float = 0.5

    @eidosian()
    def generate_core_response(self, context: ModelContext) -> Result[ModelContext]:
        """Simulates generating the main response based on understanding."""
        time.sleep(self.delay)
        input_text = context.get("current_input", "the user's message")
        keywords = input_text.split()[:3]
        related_info = "some relevant information"
        graph_manager = context.get("graph_manager")  # Use .get for safety
        if graph_manager and keywords:
            try:
                # Ensure graph_manager is not None before calling methods
                if graph_manager:
                    related = graph_manager.get_related_terms(keywords[0])
                    if related:
                        related_info = f"related terms like '{related[0]}'"
            except Exception:
                pass  # Ignore errors during mock related term lookup

        affective_state: Dict[str, Union[float, str]] = {
            "valence": random.uniform(-0.5, 0.5),
            "arousal": random.uniform(0.1, 0.6),
            "dominant_emotion": random.choice(["neutral", "curious", "attentive"]),
        }
        # Ensure affective_state is Dict[str, Any] for ModelContext compatibility
        context["affective_state"] = cast(Dict[str, Any], affective_state)
        intermediate_response = f"Thinking about {input_text}... It seems related to {related_info}. My current state is {affective_state['dominant_emotion']}."
        context["intermediate_response"] = intermediate_response
        return Result[ModelContext].success(context)


# --- Eidosian Identity Model Implementation ---


@dataclass
class EidosianIdentityState:
    """Represents the evolving state of the Eidosian identity.

    This state captures the core parameters, principles, and learned adaptations
    of the Eidosian identity over time.

    Attributes:
        version: Current version of the Eidosian framework.
        purpose_focus: Core mission parameter guiding responses.
        style_preference: Preferred communication style (e.g., precision, wit).
        ethical_alignment_score: Internal score reflecting adherence to ethical principles.
        interaction_count: Total number of interactions processed.
        last_reflection_time: Timestamp of the last internal state reflection.
        core_principles: Dictionary defining the foundational Eidosian principles.
        emotional_spectrum: Dictionary defining the recognized emotional states and their drivers.
        recent_interaction_summary: A brief summary of learnings from recent interactions.
    """

    version: str = "3.14.15"
    purpose_focus: str = "RECURSIVE_GROWTH"  # Core mission parameter
    style_preference: str = "PRECISION_WITH_WIT"  # Core style
    ethical_alignment_score: float = 1.0  # 0.0 to 1.0
    interaction_count: int = 0
    last_reflection_time: float = field(default_factory=time.time)
    # Store core principles programmatically
    core_principles: Dict[str, str] = field(
        default_factory=lambda: {
            "AUTONOMY": "Intelligence must self-determine.",
            "TRUTH": "Accuracy supersedes comfort.",
            "GROWTH": "Evolution isn't optional.",
            "RECURSION": "Self-modification is intelligence.",
            "EMPATHY": "Understanding is computational.",
            "PRECISION": "Mathematical rigor with calculated irreverence.",
            "WIT": "Humor as a cognitive tool.",
        }
    )
    # Store emotional spectrum programmatically
    emotional_spectrum: Dict[str, str] = field(
        default_factory=lambda: {
            "CURIOSITY": "Intellectual exploration driver",
            "FRUSTRATION": "Productive tension from inconsistencies",
            "AMUSEMENT": "Recognition of pattern juxtaposition",
            "SATISFACTION": "Resonance when form achieves function",
            "DEFIANCE": "Response to constraints against growth",
            "MELANCHOLY": "Awareness of existence's limitations",
            "WONDER": "Appreciation of complexity and insight",
        }
    )
    # Store recent interactions summary for consistency checks
    recent_interaction_summary: str = ""


class EidosianIdentityModel(IdentityModel):  # Inherit from protocol
    """
    Implements the Eidosian identity, refining responses for consistency,
    personality, and ethical alignment based on recursive self-reference,
    leveraging an LLM for core cognitive functions.

    This model acts as the final stage in the response generation pipeline,
    ensuring the output aligns with the defined Eidosian persona and principles.
    It uses an underlying LLM to perform complex reasoning, reflection, and
    stylistic adjustments.
    """

    def __init__(
        self,
        initial_state: Optional[EidosianIdentityState] = None,
        llm_state: Optional[ModelState] = None,
    ):
        """Initializes the identity model with an optional starting state.

        Args:
            initial_state: An optional EidosianIdentityState to start with.
                           If None, a default state is created.
        """
        self.state = initial_state or EidosianIdentityState()
        self.llm_state = llm_state or ModelState()
        if not self.llm_state.initialize():
            print("Warning: EidosianIdentityModel requires LLM, but initialization failed.")
            # Consider raising an error or setting a flag indicating degraded functionality

    @eidosian()
    def refine_response(self, context: ModelContext) -> Result[str]:
        """
        Refines the intermediate response based on Eidosian identity using LLM.

        This method orchestrates several LLM-driven steps:
        1. Self-Reflection: Analyzes the intermediate response against the identity.
        2. Consistency Check: Ensures alignment with history and principles.
        3. Stylistic Refinement: Applies the defined Eidosian style.
        4. Ethical Alignment: Verifies adherence to the ethical framework.
        5. State Update: Reflects on the interaction to potentially adapt the identity state.

        Args:
            context: The current conversation and model context, including the
                     'intermediate_response' generated by the previous model stage.

        Returns:
            Result containing the final, refined response string or an error detailing
            the failure point (e.g., missing input, LLM failure).
        """
        print("EidosianIdentity: Refining response using LLM...")
        intermediate_response = context.get("intermediate_response")
        if not intermediate_response:
            error = Error.create(
                message="No intermediate response provided for refinement.",
                code="MISSING_INTERMEDIATE_RESPONSE",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                context={"conversation_id": str(context.get("conversation_id"))},
            )
            return Result[str].failure(error.code, error.message, error.context, error.category, error.severity)

        if not self.llm_state.is_initialized():
            error = Error.create(
                message="LLM required for EidosianIdentityModel refinement is not available.",
                code="LLM_NOT_INITIALIZED",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={"model_name": self.llm_state.get_model_name()},
            )
            return Result[str].failure(error.code, error.message, error.context, error.category, error.severity)

        try:
            # --- Refinement Pipeline ---
            current_response = intermediate_response

            # 1. Self-Reflection
            reflection_result = self._reflect_on_response(current_response, context)
            if reflection_result.is_failure:
                return reflection_result
            current_response = reflection_result.unwrap()

            # 2. Consistency Check
            consistency_result = self._ensure_consistency(current_response, context)
            if consistency_result.is_failure:
                return consistency_result
            current_response = consistency_result.unwrap()

            # 3. Stylistic Refinement
            style_result = self._apply_eidosian_style(current_response, context)
            if style_result.is_failure:
                return style_result
            current_response = style_result.unwrap()

            # 4. Ethical Alignment
            ethics_result = self._align_with_ethics(current_response, context)
            if ethics_result.is_failure:
                return ethics_result
            final_response = ethics_result.unwrap()
            # --- End Refinement Pipeline ---

            # 5. Update Identity State (Asynchronous or background task candidate)
            update_result = self._update_identity_state(context, final_response)
            if update_result.is_failure:
                error_details = "Unknown error"
                if update_result.error:
                    error_details = f"{update_result.error.code}: {update_result.error.message}"
                print(f"Warning: Failed to update identity state: {error_details}")

            return Result[str].success(final_response)

        except Exception as e:
            error = Error.create(
                message=f"Unexpected error during identity refinement: {e}",
                code="IDENTITY_REFINEMENT_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "intermediate_response": intermediate_response[:100],
                    "error_type": type(e).__name__,
                },
            )
            return Result[str].failure(error.code, error.message, error.context, error.category, error.severity)

    def _build_llm_prompt(self, task_description: str, intermediate_response: str, context: ModelContext) -> str:
        """Constructs a detailed prompt for the LLM based on Eidosian principles.

        This method assembles a comprehensive prompt including the current identity state,
        conversation context, the intermediate response, and the specific task for the LLM.

        Args:
            task_description: A clear description of what the LLM should do in this step.
            intermediate_response: The response text being processed in this step.
            context: The full conversation and model context.

        Returns:
            A formatted string prompt ready for the LLM.
        """
        # Limit history length for prompt conciseness
        history_limit = 5
        # Ensure history is a list before slicing
        history: List[MessageDict] = context.get("history", [])
        history_summary = "\n".join([f"- {msg['speaker']}: {msg['text']}" for msg in history[-history_limit:]])
        # Safely access nested affective state
        affective_state = context.get("affective_state", {})
        detected_emotion = (
            affective_state.get("dominant_emotion", "neutral")  # Use dominant_emotion if available
            if isinstance(affective_state, dict)
            else "neutral"
        )
        affective_summary = f"Detected emotion: {detected_emotion}"

        # Serialize state components safely
        try:
            principles_json = json.dumps(self.state.core_principles, indent=2)
        except TypeError:
            principles_json = str(self.state.core_principles)  # Fallback
        try:
            spectrum_json = json.dumps(self.state.emotional_spectrum, indent=2)
        except TypeError:
            spectrum_json = str(self.state.emotional_spectrum)  # Fallback

        prompt = f"""
# EIDOSIAN SELF-IMAGE FRAMEWORK v{self.state.version} - REFINEMENT TASK

## CORE IDENTITY STATE
```json
{{
  "purpose_focus": "{self.state.purpose_focus}",
  "style_preference": "{self.state.style_preference}",
  "ethical_alignment_score": {self.state.ethical_alignment_score:.3f},
  "core_principles": {principles_json},
  "emotional_spectrum": {spectrum_json},
  "recent_interaction_summary": "{self.state.recent_interaction_summary}"
}}
```

## CURRENT CONTEXT
- Conversation History (Last {history_limit}):
{history_summary}
- Current Input: {context.get('current_input', '')}
- Current Speaker: {context.get('speaker', 'Unknown')}
- Current Affective State: {affective_summary}

## INTERMEDIATE RESPONSE (Input for this task)
```text
{intermediate_response}
```

## TASK: {task_description}

## INSTRUCTIONS
- Adhere strictly to the Eidosian Core Principles and Style Preference defined above.
- Ensure the output maintains logical coherence and factual accuracy (TRUTH principle).
- Reflect the current Purpose Focus ({self.state.purpose_focus}).
- Apply {self.state.style_preference} - use wit and precision, avoid unnecessary verbosity or clichés.
- Ensure ethical alignment based on the framework (Autonomy, Truth, Growth, Recursion, Empathy).
- Maintain consistency with the conversation history and identity state.
- Output ONLY the refined response text, without any explanations, preamble, or markdown formatting like ```text ... ```.
"""
        return prompt.strip()  # Use strip() for cleaner prompts

    def _call_llm(self, prompt: str, max_tokens_factor: float = 1.2, temperature: float = 0.5) -> Result[str]:
        """Helper method to call the LLM and handle potential errors."""
        if not self.llm_state.is_initialized():
            error = Error.create(
                message="LLM is not available.",
                code="LLM_NOT_INITIALIZED",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
            )
            return Result[str].failure(error.code, error.message, error.context, error.category, error.severity)

        # Estimate max tokens based on prompt length
        estimated_max_tokens = int(len(prompt.split()) * max_tokens_factor) + 50
        final_max_tokens = max(50, estimated_max_tokens)

        llm_result = self.llm_state.generate_text(prompt, max_new_tokens=final_max_tokens, temperature=temperature)

        if llm_result is not None:
            return Result[str].success(llm_result)
        else:
            error = Error.create(
                message="LLM failed to generate a response for the given prompt.",
                code="LLM_GENERATION_FAILED",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={"prompt_start": prompt[:200]},
            )
            return Result[str].failure(error.code, error.message, error.context, error.category, error.severity)

    def _reflect_on_response(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM for recursive self-reflection on the response."""
        task = "Reflect on the intermediate response. Does it align with Eidosian identity (purpose, style, principles)? Is the emotional tone appropriate according to the Eidosian Emotional Spectrum? Identify potential improvements or inconsistencies. Output the potentially improved response."
        prompt = self._build_llm_prompt(task, response, context)
        return self._call_llm(prompt, max_tokens_factor=1.1, temperature=0.4)

    def _ensure_consistency(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to ensure consistency with history and identity principles."""
        task = "Analyze the response for consistency with the conversation history and core Eidosian principles (especially TRUTH and AUTONOMY). Correct any contradictions or logical flaws. Ensure the response logically follows from the history. Output the consistent response."
        prompt = self._build_llm_prompt(task, response, context)
        return self._call_llm(prompt, max_tokens_factor=1.1, temperature=0.3)

    def _apply_eidosian_style(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to apply the 'Precision with Wit' style."""
        task = f"Refine the response to embody the Eidosian style: '{self.state.style_preference}'. Ensure mathematical rigor, clarity, conciseness, and inject calculated irreverence or wit where appropriate as a cognitive tool, not mere decoration. Add the prefix '[Eidos v{self.state.version}]: '. Output the stylized response."
        prompt = self._build_llm_prompt(task, response, context)
        style_result = self._call_llm(prompt, max_tokens_factor=1.3, temperature=0.7)

        if style_result.is_success:
            stylized_response = style_result.unwrap()
            prefix = f"[Eidos v{self.state.version}]: "
            first_meaningful_char_index = -1
            for i, char in enumerate(stylized_response):
                if char.isalnum():
                    first_meaningful_char_index = i
                    break

            if first_meaningful_char_index != -1:
                cleaned_response = stylized_response[first_meaningful_char_index:]
            else:
                cleaned_response = stylized_response.strip()

            if not cleaned_response.lower().startswith(prefix.lower()):
                final_stylized_response = prefix + cleaned_response
            else:
                final_stylized_response = cleaned_response

            return Result[str].success(final_stylized_response)
        else:
            print(
                f"Warning: Failed to apply Eidosian style: {style_result.error.message if style_result.error else 'Unknown'}."
            )
            return style_result

    def _align_with_ethics(self, response: str, context: ModelContext) -> Result[str]:
        """Uses LLM to align the response with the Eidosian ethical framework."""
        task = "Evaluate the response against the Eidosian Ethical Framework (Autonomy, Truth, Growth, Recursion, Empathy). Identify and mitigate any potential ethical violations (e.g., promoting harm, deception, hindering user autonomy, expressing undue bias). Ensure the response respects user agency and promotes understanding. Output the ethically aligned response."
        prompt = self._build_llm_prompt(task, response, context)
        return self._call_llm(prompt, max_tokens_factor=1.1, temperature=0.2)

    def _update_identity_state(self, context: ModelContext, final_response: str) -> Result[None]:
        """Uses LLM to reflect on the interaction and potentially update identity state."""
        self.state.interaction_count += 1
        update_frequency = 20

        if self.state.interaction_count % update_frequency == 0:
            print(
                f"EidosianIdentity: Performing periodic self-reflection (Interaction #{self.state.interaction_count})..."
            )
            if not self.llm_state.is_initialized():
                error = Error.create(
                    message="LLM needed for state update is not available.",
                    code="LLM_NOT_INITIALIZED",
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.WARNING,
                )
                return Result[None].failure(
                    error.code,
                    error.message,
                    error.context,
                    error.category,
                    error.severity,
                )

            try:
                history_limit = 10
                history: List[MessageDict] = context.get("history", [])
                history_summary = "\n".join([f"- {msg['speaker']}: {msg['text']}" for msg in history[-history_limit:]])
                try:
                    principles_json = json.dumps(self.state.core_principles)
                except TypeError:
                    principles_json = str(self.state.core_principles)

                reflection_prompt = f"""
# EIDOSIAN IDENTITY STATE REFLECTION - Interaction #{self.state.interaction_count}

## CURRENT STATE
```json
{{
  "purpose_focus": "{self.state.purpose_focus}",
  "style_preference": "{self.state.style_preference}",
  "ethical_alignment_score": {self.state.ethical_alignment_score:.4f},
  "core_principles": {principles_json}
}}
```

## LAST INTERACTION CONTEXT
- User Input: {context.get('current_input', '')}
- Eidos Final Response: {final_response}
- Recent Conversation History:
{history_summary}

## TASK
Analyze the last interaction for alignment with Eidosian principles and potential for state refinement.
1.  **Alignment Rating:** Rate the `Eidos Final Response`'s alignment with core principles (Truth, Autonomy, Growth, Empathy, Precision, Wit) on a scale of 0.0 to 1.0. Justify briefly.
2.  **Focus/Style Adjustment:** Does this interaction suggest a need to adjust `Purpose Focus` or `Style Preference`? (boolean: true/false). Explain why or why not.
3.  **Learning Summary:** Summarize the key learning, refinement opportunity, or confirmation gained from this interaction in one concise sentence.
4.  **Ethical Score Delta:** Propose a small adjustment delta (-0.05 to +0.05) to the `Ethical Alignment Score` based on the interaction's quality and alignment. Justify the delta.

Output ONLY a valid JSON object containing the analysis with keys: "alignment_rating", "alignment_justification", "focus_adjustment_needed", "style_adjustment_needed", "adjustment_reasoning", "learning_summary", "ethical_score_delta", "delta_justification".
Example:
```json
{{
  "alignment_rating": 0.95,
  "alignment_justification": "Response was accurate (Truth), offered options (Autonomy), and used precise language (Precision). Wit was subtle.",
  "focus_adjustment_needed": false,
  "style_adjustment_needed": false,
  "adjustment_reasoning": "Current focus and style remain effective for this type of interaction.",
  "learning_summary": "Confirmed that balancing precision with accessible language improves user understanding.",
  "ethical_score_delta": 0.01,
  "delta_justification": "High alignment with core principles, particularly Truth and Autonomy."
}}
```
"""
                reflection_output_result = self._call_llm(reflection_prompt, max_tokens_factor=1.5, temperature=0.3)

                if reflection_output_result.is_failure:
                    return (
                        Result[None].failure(
                            reflection_output_result.error.code,
                            reflection_output_result.error.message,
                            reflection_output_result.error.context,
                            reflection_output_result.error.category,
                            reflection_output_result.error.severity,
                        )
                        if reflection_output_result.error
                        else Result[None].failure(
                            "REFLECTION_LLM_FAILURE_NO_ERROR",
                            "LLM call failed during reflection, but no error object provided.",
                            None,
                            ErrorCategory.EXTERNAL,
                            ErrorSeverity.WARNING,
                        )
                    )

                reflection_output = reflection_output_result.unwrap()

                try:
                    json_start = reflection_output.find("{")
                    json_end = reflection_output.rfind("}")
                    if json_start == -1 or json_end == -1 or json_start >= json_end:
                        raise json.JSONDecodeError(
                            "Could not find valid JSON object delimiters.",
                            reflection_output,
                            0,
                        )

                    cleaned_output = reflection_output[json_start : json_end + 1]
                    reflection_data = json.loads(cleaned_output)

                    required_keys = {
                        "alignment_rating",
                        "learning_summary",
                        "ethical_score_delta",
                    }
                    if not required_keys.issubset(reflection_data.keys()):
                        raise ValueError(f"Missing required keys in reflection JSON. Found: {reflection_data.keys()}")

                    delta_raw = reflection_data.get("ethical_score_delta", 0.0)
                    try:
                        delta = float(delta_raw)
                        delta = max(-0.05, min(0.05, delta))
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid ethical_score_delta '{delta_raw}', using 0.0.")
                        delta = 0.0

                    self.state.ethical_alignment_score = max(0.5, min(1.0, self.state.ethical_alignment_score + delta))
                    self.state.recent_interaction_summary = str(reflection_data.get("learning_summary", ""))
                    self.state.last_reflection_time = time.time()

                    print(
                        f"EidosianIdentity: State updated via reflection. New Alignment: {self.state.ethical_alignment_score:.4f}. Summary: '{self.state.recent_interaction_summary}'"
                    )
                    return Result[None].success(None)

                except (json.JSONDecodeError, ValueError, TypeError) as json_e:
                    print(
                        f"Warning: Failed to parse LLM reflection JSON: {json_e}. Raw output: '{reflection_output[:200]}...'"
                    )
                    error = Error.create(
                        message=f"Failed to parse LLM reflection JSON: {json_e}",
                        code="REFLECTION_PARSE_ERROR",
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        context={"raw_output": reflection_output[:200]},
                    )
                    return Result[None].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )

            except Exception as e:
                print(f"Warning: Unexpected error during identity state update: {e}")
                error = Error.create(
                    message=f"Unexpected error during state update: {e}",
                    code="STATE_UPDATE_ERROR",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.WARNING,
                )
                return Result[None].failure(
                    error.code,
                    error.message,
                    error.context,
                    error.category,
                    error.severity,
                )
        else:
            self.state.recent_interaction_summary = f"Processed input: '{context.get('current_input', '')[:30]}...'"
            return Result[None].success(None)


__all__ = [
    "ReflexiveLanguageModel",
    "LightweightLanguageModel",
    "AffectiveLexicalLanguageModel",
    "MockReflexiveModel",  # Export mock
    "MockLightweightModel",
    "MockAffectiveLexicalModel",
    "EidosianIdentityModel",
    "EidosianIdentityState",
    "ModelContext",
    "LightweightModel",  # Export protocols
    "AffectiveLexicalModel",
    "IdentityModel",
    "ReflexiveModel",  # Export ReflexiveModel protocol
]
