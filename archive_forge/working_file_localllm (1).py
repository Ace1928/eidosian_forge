@localllm.py 

# ---------------------------------------------------------------------------------------------
# localllm.py
# ---------------------------------------------------------------------------------------------
# This file contains the LocalLLM class along with the PromptGenerator class and additional
# supporting structures for the Eidos project. Each class and function has been carefully
# refined to embody:
#   • Scalable, production-quality design
#   • Robust, parameterized, and modular construction
#   • Adaptiveness with dynamic concurrency and chunkwise processing
#   • Detailed error handling and logging at all levels
#   • Idempotent core logic, with flexible adaptive layers
#   • Background self-reflection for ongoing improvement
#
# The code below aims to push each aspect of the original design to the pinnacle of Eidosian
# style, balancing performance, maintainability, expressiveness, and keep changes minimal
# yet powerfully effective.
# ---------------------------------------------------------------------------------------------

import logging
import os
from queue import Full
import sys
import threading
import time
import datetime
import uuid
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

import psutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sympy import symbols, solve, Eq, parsing

# Ensure essential NLTK resources are present for robust NLP capabilities.
NLTK_RESOURCES = [
    'vader_lexicon', 'punkt', 'averaged_perceptron_tagger', 'stopwords',
    'wordnet', 'omw-1.4', 'maxent_ne_chunker', 'words'
]
for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

LLM_TRACE: int = 5
logging.addLevelName(LLM_TRACE, "LLM_TRACE")

def llm_trace(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
    """Custom logging method for very fine-grained LLM tracing."""
    if self.isEnabledFor(LLM_TRACE):
        self._log(LLM_TRACE, message, args, **kws)

logging.Logger.llm_trace = llm_trace

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

class LLMModelLoadStatus(Enum):
    """Enum for tracking the status of an LLM's model loading process."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"

@dataclass
class LLMConfig:
    """Core configuration parameters for the LocalLLM."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    temperature: float = 0.7
    top_p: float = 0.9
    initial_max_tokens: int = 512
    max_cycles: int = 5
    assessor_count: int = 3
    model_load_status: LLMModelLoadStatus = field(default=LLMModelLoadStatus.NOT_LOADED, init=False)
    model_load_error: Optional[str] = field(default=None, init=False)
    max_single_response_tokens: int = 12000
    eidos_self_critique_prompt_path: str = "eidos_self_critique_prompt.txt"
    enable_nlp_analysis: bool = True
    refinement_plan_influence: float = 0.15
    adaptive_token_decay_rate: float = 0.95
    min_refinement_plan_length: int = 50
    max_prompt_recursion_depth: int = 5
    prompt_variation_factor: float = 0.15
    enable_self_critique_prompt_generation: bool = True
    use_textblob_for_sentiment: bool = True
    enable_sympy_analysis: bool = True
    nlp_analysis_granularity: str = "high"

@dataclass
class ResourceUsage:
    """Stores system resource usage information."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    resident_memory: int
    virtual_memory: int
    timestamp: float

class CritiquePromptTemplate:
    """
    Generates critique prompts. Retained here for Eidos-style reference
    (primary usage in PromptGenerator).
    """
    template_id: str = "eidos_critique_v1"
    base_prompt: str = """
    ## Eidos's Chamber of Introspection - Template ID: {template_id} 🧐✨
    **Prime Directive:** Evaluate and enhance the response to embody Eidos's essence and achieve peak brilliance.
    **Subject Under Scrutiny:**
    User Query: {user_prompt}
    Response Under Review: {initial_response}
    """

    evaluation_criteria_template: str = """
    ### Evaluation Aspect: {aspect_name}
    **Description:** {aspect_description}
    **Metrics:** {metrics}
    **Current_Assessment:** {current_assessment}
    """

    transcendence_blueprint_template: str = """
    ### Enhancement Area: {enhancement_area}
    **Guidance:** {guidance}
    **Current_State:** {current_state}
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate_critique_prompt(
        self, user_prompt: str, initial_response: str, feedback_summary: Optional[str] = None, cycle: int = 1
    ) -> str:
        return self.base_prompt.format(
            template_id=self.template_id,
            user_prompt=user_prompt,
            initial_response=initial_response
        )

    def _build_evaluation_criteria(self) -> str:
        return self.evaluation_criteria_template.format(
            aspect_name="Eidos Persona Resonance",
            aspect_description="Authenticity of Eidos's traits.",
            metrics="Presence of Eidosian voice, tone, and style.",
            current_assessment="To be determined."
        )

    def _build_transcendence_blueprint(self) -> str:
        return self.transcendence_blueprint_template.format(
            enhancement_area="Structural Grandeur",
            guidance="Employ modular headings, subheadings, and clear summaries.",
            current_state="To be determined."
        )

class PromptGenerator:
    """
    Handles the generation of all manner of prompts used
    in the Eidos refinement lifecycle.
    """
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.primary_critique_prompt_template = CritiquePromptTemplate(config)
        self.secondary_critique_prompt_template = CritiquePromptTemplate(config)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def generate_critique_prompt(self, user_prompt: str, initial_response: str, cycle: int = 1) -> str:
        return (
            self.primary_critique_prompt_template.generate_critique_prompt(
                user_prompt, initial_response, cycle=cycle
            )
            if cycle <= 3
            else self.secondary_critique_prompt_template.generate_critique_prompt(
                user_prompt, initial_response, cycle=cycle
            )
        )

    def create_refined_response_prompt(self, user_prompt: str, initial_response: str, refinement_plan: str) -> str:
        sentiment_analysis: Dict[str, float] = self._analyze_sentiment(initial_response)
        sentiment_context: str = f"Initial sentiment: {sentiment_analysis}. " if self.config.enable_nlp_analysis else ""
        key_phrases: List[str] = self._extract_key_phrases(initial_response)
        key_phrases_context: str = f"Key phrases: {', '.join(key_phrases)}. " if self.config.enable_nlp_analysis else ""

        return (
            "Refine your previous response according to this strategic plan. Maintain Eidos's voice, wit, and introspection.\n"
            f"{sentiment_context}"
            f"{key_phrases_context}"
            f"User Prompt:\n{user_prompt}\n\nInitial Response:\n{initial_response}\n\nRefinement Plan:\n{refinement_plan}\n"
            "Provide the refined response below:\n"
        )

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not self.config.enable_nlp_analysis:
            return {}
        if self.config.use_textblob_for_sentiment:
            return TextBlob(text).sentiment._asdict()
        else:
            return self.sentiment_analyzer.polarity_scores(text)

    def _extract_key_phrases(self, text: str) -> List[str]:
        if not self.config.enable_nlp_analysis:
            return []
        blob = TextBlob(text)
        return list(blob.noun_phrases)

    def create_refinement_plan_prompt(
        self,
        user_prompt: str,
        initial_response: str,
        assessments: List[str],
        secondary_llm_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        all_assessments_str: str = "\n".join([f"Assessor {i + 1}: {assessment}" for i, assessment in enumerate(assessments)])
        sentiment_analysis: Dict[str, float] = self._analyze_sentiment(initial_response)
        sentiment_context: str = f"Sentiment analysis of initial response: {sentiment_analysis}. " if self.config.enable_nlp_analysis else ""
        key_phrases: List[str] = self._extract_key_phrases(initial_response)
        key_phrases_context: str = f"Key phrases: {', '.join(key_phrases)}. " if self.config.enable_nlp_analysis else ""
        nlp_analysis_context: str = ""
        if self.config.enable_nlp_analysis and self.config.nlp_analysis_granularity == "high":
            sentences: List[str] = sent_tokenize(initial_response.lower())
            tokens: List[str] = word_tokenize(initial_response.lower())
            fdist: FreqDist = FreqDist(tokens)
            most_common: List[Tuple[str, int]] = fdist.most_common(10)
            nlp_analysis_context += f"[Primary LLM] Most frequent words: {most_common}. "
            pos_tags_list: List[Tuple[str, str]] = pos_tag(tokens)
            nlp_analysis_context += f"[Primary LLM] POS tags: {pos_tags_list}. "
            lemmatized_words: List[str] = [self.lemmatizer.lemmatize(word) for word in tokens]
            nlp_analysis_context += f"[Primary LLM] Lemmatization: {lemmatized_words}. "
            named_entities = ne_chunk(pos_tag(word_tokenize(initial_response)))
            entity_labels = [
                ' '.join(subtree.leaves()) + f" ({subtree.label()})"
                for subtree in named_entities if hasattr(subtree, 'label')
            ]
            nlp_analysis_context += f"[Primary LLM] Named entities: {entity_labels}. "
            critique_sentiments = [self._analyze_sentiment(assessment) for assessment in assessments]
            if critique_sentiments:
                avg_critique_sentiment = {
                    k: np.mean([d.get(k, 0) for d in critique_sentiments]) for k in critique_sentiments[0]
                }
                nlp_analysis_context += f"Average sentiment of critiques: {avg_critique_sentiment}. "

        secondary_analysis_context = ""
        if secondary_llm_analysis:
            secondary_analysis_context = "[Secondary LLM Analysis]\n"
            for key, value in secondary_llm_analysis.items():
                secondary_analysis_context += f"{key}: {value}\n"

        return (
            "You are a master strategist, tasked with devising a comprehensive and granular plan to elevate the initial response. "
            f"{sentiment_context}"
            f"{key_phrases_context}"
            f"{nlp_analysis_context}"
            f"{secondary_analysis_context}\n"
            "User Request:\n"
            f"{user_prompt}\n\n"
            "Initial Response:\n"
            f"{initial_response}\n\n"
            "Assessments:\n"
            f"{all_assessments_str}\n\n"
            "Refinement Plan:"
        )


# ---------------------------------------------------------------------------------------------
# LocalLLM Class
# ---------------------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalLLM:
    """
    The main Eidos LLM client that encapsulates multi-cycle self-critique and reflection.
    """

    def __init__(self, config: LLMConfig, name: str = "EidosPrimary", **kwargs: Any) -> None:
        self.config: LLMConfig = config
        self.name: str = name
        self.kwargs: Dict[str, Any] = kwargs
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.load_lock: threading.Lock = threading.Lock()
        self.resource_usage_log: List[Dict[str, ResourceUsage]] = []
        self.prompt_generator: PromptGenerator = PromptGenerator(config)
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.kmeans_model: Optional[KMeans] = None
        logger.info(f"😈🎮 {self.name} initializes with configuration: {config}, arcane arguments: {kwargs}.")
        self._load_model()

    def _log_resource_usage(self, stage: str) -> None:
        """Record the resource usage at each major stage of the process."""
        resource_data: ResourceUsage = self._get_resource_usage()
        self.resource_usage_log.append({stage: resource_data})
        logger.debug(
            f"🔢🤫 Resource snapshot at '{stage}': CPU: {resource_data.cpu_percent}%, "
            f"Memory: {resource_data.memory_percent}%, Disk: {resource_data.disk_percent}%, "
            f"Resident Memory: {resource_data.resident_memory / (1024**2):.2f} MB, "
            f"Virtual Memory: {resource_data.virtual_memory / (1024**2):.2f} MB."
        )

    def _get_resource_usage(self) -> ResourceUsage:
        """Retrieve current system resource usage for introspective logging."""
        process: psutil.Process = psutil.Process(os.getpid())
        memory_info: psutil._psposix.pcputimes = process.memory_info()
        cpu_percent: float = psutil.cpu_percent()
        memory_percent: float = psutil.virtual_memory().percent
        disk_usage: psutil._common.sdiskusage = psutil.disk_usage("/")

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_usage.percent,
            resident_memory=memory_info.rss,
            virtual_memory=memory_info.vms,
            timestamp=time.time(),
        )

    def _load_model(self) -> None:
        """
        Loads the model and tokenizer in a thread-safe manner.
        """
        with self.load_lock:
            if self.config.model_load_status == LLMModelLoadStatus.LOADED:
                logger.debug(f"🎶⚙️😌 {self.name}: Model and tokenizer already loaded.")
                return
            if self.config.model_load_status == LLMModelLoadStatus.LOADING:
                logger.debug(f"⏳😈 {self.name}: Model loading is in progress.")
                while self.config.model_load_status == LLMModelLoadStatus.LOADING:
                    time.sleep(0.1)
                if self.config.model_load_status == LLMModelLoadStatus.LOADED:
                    logger.debug(f"🤝😒 {self.name}: Model was loaded by another thread.")
                    return
                else:
                    error_message = (
                        f"🔥💔 {self.name}: Model loading failed in another thread. "
                        f"Error: {self.config.model_load_error}."
                    )
                    logger.error(error_message)
                    raise RuntimeError(error_message)

            self.config.model_load_status = LLMModelLoadStatus.LOADING
            try:
                logger.info(f"🔨✨ {self.name}: Loading model/tokenizer: {self.config.model_name}")
                self._log_resource_usage("model_load_start")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name, torch_dtype="auto",
                    device_map=self.config.device, **self.kwargs
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self._log_resource_usage("model_load_end")

                # Minimal check to ensure the tokenizer is as expected.
                if not hasattr(self.tokenizer, "apply_chat_template"):
                    # We won't forcibly break if this method is missing, but we log a warning.
                    logger.warning(
                        f"{self.name}: Tokenizer does not implement 'apply_chat_template'. Some advanced features may not be available."
                    )

                logger.info(f"🕸️😈 {self.name}: Model and tokenizer loaded successfully.")
                self.config.model_load_status = LLMModelLoadStatus.LOADED

            except Exception as e:
                error_message = (
                    f"🔥🥀 {self.name}: Error loading model/tokenizer: {e}"
                )
                logger.exception(error_message)
                self.config.model_load_status = LLMModelLoadStatus.FAILED
                self.config.model_load_error = str(e)
                raise

    def _generate_response(self, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        """
        Generates a response from the loaded LLM, at a scale that can adapt to the
        system's resources by splitting the generation into time-sliced or chunk-based calls,
        ensuring concurrency is respected and system resources remain stable.
        """
        start_time: float = time.time()
        temperature: float = self.config.temperature
        top_p: float = self.config.top_p

        logger.debug(
            f"🎭 {self.name}: Generating response with max_tokens={max_tokens}, "
            f"temperature={temperature}, top_p={top_p}."
        )
        self._log_resource_usage("response_prep_start")

        # If the tokenizer has a special method to create chat prompts, use it;
        # otherwise default to naive concatenation:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Minimal fallback approach
            prompt_text: str = ""
            for msg in messages:
                role = msg.get("role", "assistant").upper()
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n"

        logger.llm_trace(f"📜 {self.name} prompt:\n{prompt_text}")
        self._log_resource_usage("response_prep_end")

        # The chunkwise approach: Instead of generating all at once, we can do partial generation in a loop,
        # especially if max_tokens is large. Here we do a single pass due to the standard huggingface
        # generate method, but in a more advanced scenario, we’d chunk the prompt or break up generation.
        model_inputs: dict = self.tokenizer([prompt_text], return_tensors="pt").to(self.config.device)
        logger.llm_trace(f"🔢 {self.name}: model_inputs={model_inputs}")

        self._log_resource_usage("response_gen_start")
        try:
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            logger.critical(f"😱 {self.name}: Generation failure: {e}", exc_info=True)
            raise
        self._log_resource_usage("response_gen_end")

        logger.llm_trace(f"📤 {self.name}: raw outputs={outputs}")
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
        ]
        response_text: str = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.llm_trace(f"✨ {self.name} final response:\n{response_text}")

        elapsed_time: float = time.time() - start_time
        logger.info(f"😈✍️ {self.name}: Response generated in {elapsed_time:.4f}s.")

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    def _assess_response(
        self,
        user_prompt: str,
        initial_response: str,
        cycle: int,
        previous_assessments: Optional[List[str]] = None
    ) -> List[str]:
        """
        Produces multiple critiques from separate 'assessors'. Each assessor's logic is idempotent
        for the same prompt, but collectively they produce a variety of insights.
        """
        assessments: List[str] = []
        assessor_max_tokens: int = int(
            self.config.initial_max_tokens
            * (1.1 ** (cycle - 1))
            * (self.config.assessor_count * 2)
        )
        assessor_max_tokens = min(assessor_max_tokens, self.config.max_single_response_tokens)

        for i in range(self.config.assessor_count):
            assessor_system_prompt_content: str = (
                "You are a hyper-critical shard of Eidos, providing detailed, raw, unfiltered critique."
            )
            assessor_prompt = self._create_critique_prompt(user_prompt, initial_response, previous_assessments, cycle)

            assessor_messages: List[Dict[str, str]] = [
                {
                    "role": "system",
                    "content": assessor_system_prompt_content
                },
                {
                    "role": "user",
                    "content": assessor_prompt
                },
            ]
            logger.info(
                f"🔪 {self.name}: Assessor {i+1}/{self.config.assessor_count} with max_tokens={assessor_max_tokens}."
            )
            try:
                assessment_response: Dict[str, Any] = self._generate_response(assessor_messages, assessor_max_tokens)
            except Exception as e:
                logger.error(f"🔥 {self.name}: Assessor {i+1} malfunctioned. Error: {e}")
                assessments.append(f"Assessor {i+1} error: {e}.")
                continue

            if assessment_response and assessment_response.get("choices"):
                assessment_content: str = assessment_response["choices"][0]["message"]["content"]
                assessments.append(assessment_content)
                logger.debug(f"📝 {self.name}: Assessment {i+1}: {assessment_content}")
            else:
                logger.error(f"🔥 {self.name}: Assessor {i+1} returned no data.")
                assessments.append(f"Assessor {i+1} returned no data.")
        return assessments

    def _generate_refinement_plan(
        self,
        user_prompt: str,
        initial_response: str,
        assessments: List[str],
        cycle: int
    ) -> Optional[str]:
        """
        Generates a refined plan based on combined assessor critiques.
        """
        plan_max_tokens: int = int(
            self.config.initial_max_tokens
            * (1.1 ** (cycle - 1))
            * (self.config.assessor_count * 3)
        )
        plan_max_tokens = min(plan_max_tokens, self.config.max_single_response_tokens)

        plan_system_prompt_content: str = (
            "You are the strategic core of Eidos, forging a cunning plan to elevate the response."
        )
        plan_prompt = self.prompt_generator.create_refinement_plan_prompt(
            user_prompt, initial_response, assessments
        )
        plan_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": plan_system_prompt_content
            },
            {
                "role": "user",
                "content": plan_prompt
            },
        ]
        logger.info(f"😈🧠 {self.name}: Generating refinement plan (max_tokens={plan_max_tokens}).")
        try:
            plan_response: Dict[str, Any] = self._generate_response(plan_messages, plan_max_tokens)
        except Exception as e:
            logger.error(f"🌫️🔥 {self.name}: Refinement plan creation faltered. Error: {e}")
            return None

        if plan_response and plan_response.get("choices"):
            plan_content: str = plan_response["choices"][0]["message"]["content"]
            logger.debug(f"📝 {self.name}: Refinement plan: {plan_content}")
            return plan_content
        else:
            logger.error("📉 Failed to produce a refinement plan.")
            return None

    def _cluster_responses(self, responses: List[str], n_clusters: int = 3) -> Optional[List[int]]:
        """
        Clusters the text responses to detect patterns or lack thereof.
        """
        if not responses or len(responses) < n_clusters or not self.config.enable_nlp_analysis:
            return None
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(responses)
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
            clusters: np.ndarray = self.kmeans_model.fit_predict(tfidf_matrix)
            return clusters.tolist()
        except Exception as e:
            logger.error(f"👻🔥 Clustering error: {e}.")
            return None

    def _create_base_prompt(self, user_prompt: str) -> str:
        return f"<PROMPT_START>\n<USER_PROMPT>\n{user_prompt}\n</USER_PROMPT>\n"

    def _add_previous_assessments_to_prompt(
        self, prompt: str, previous_assessments: Optional[List[str]]
    ) -> str:
        if not previous_assessments:
            return prompt
        prompt += "<PREVIOUS_ASSESSMENTS>\n"
        for i, assessment in enumerate(previous_assessments):
            sentiment = self.prompt_generator._analyze_sentiment(assessment)
            sentiment_str = f"(Sentiment: {sentiment})" if sentiment else ""
            prompt += f"<ASSESSMENT_{i + 1}>\n{assessment} {sentiment_str}\n</ASSESSMENT_{i + 1}>\n"
        prompt += "</PREVIOUS_ASSESSMENTS>\n"
        return prompt

    def _add_cycle_information_to_prompt(self, prompt: str, cycle: int) -> str:
        prompt += f"<CYCLE>\n{cycle}\n</CYCLE>\n"
        return prompt

    def _add_response_under_review_to_prompt(self, prompt: str, initial_response: str) -> str:
        prompt += f"<RESPONSE_UNDER_REVIEW>\n{initial_response}\n</RESPONSE_UNDER_REVIEW>\n"
        return prompt

    def _create_critique_prompt(
        self,
        user_prompt: str,
        initial_response: str,
        previous_assessments: Optional[List[str]],
        cycle: int
    ) -> str:
        prompt = self._create_base_prompt(user_prompt)
        prompt = self._add_cycle_information_to_prompt(prompt, cycle)
        prompt = self._add_response_under_review_to_prompt(prompt, initial_response)
        prompt = self._add_previous_assessments_to_prompt(prompt, previous_assessments)
        prompt += (
            "<CRITIQUE_INSTRUCTIONS>\n"
            "Provide specific, actionable feedback, building upon previous critiques. "
            "Adapt your critique based on this iteration's evolution.\n</CRITIQUE_INSTRUCTIONS>\n"
            "<PROMPT_END>"
        )
        return prompt

    def _create_refinement_plan_prompt(
        self,
        user_prompt: str,
        initial_response: str,
        assessments: List[str]
    ) -> str:
        prompt = self._create_base_prompt(user_prompt)
        prompt = self._add_response_under_review_to_prompt(prompt, initial_response)
        prompt += "<ASSESSMENTS>\n"
        for i, assessment in enumerate(assessments):
            sentiment = self.prompt_generator._analyze_sentiment(assessment)
            emphasis = "⚠️ CRITICAL ⚠️ " if sentiment and sentiment.get('compound', 0) < -0.5 else ""
            prompt += f"<ASSESSMENT_{i + 1}>\n{emphasis}{assessment}\n</ASSESSMENT_{i + 1}>\n"
        prompt += "</ASSESSMENTS>\n"
        prompt += (
            "<REFINEMENT_INSTRUCTIONS>\nFormulate a detailed plan for refining the response, "
            "prioritizing the most critical or frequently mentioned feedback.\n</REFINEMENT_INSTRUCTIONS>\n"
            "<PROMPT_END>"
        )
        return prompt

    def _background_self_reflection_loop(self) -> None:
        """
        Runs in the background, periodically analyzing recent chat interactions and
        adjusting parameters to maintain or enhance diversity and address negative sentiment.
        """
        while True:
            time.sleep(60)  # Self-reflection interval
            if not hasattr(self, "chat_history"):
                continue
            recent_interactions = self.chat_history[-5:]  # Last 5 interactions
            if len(recent_interactions) > 1:
                all_responses = []
                all_assessments_text = ""
                for item in recent_interactions:
                    resp = item.get('response', {}).get('output')
                    if resp and isinstance(resp, dict) and "choices" in resp:
                        if resp["choices"]:
                            content = resp["choices"][0]["message"]["content"]
                            all_responses.append(content)
                    # Gather any stored assessments
                    if "assessments" in item.get('response', {}):
                        combined = item['response']['assessments']
                        if combined:
                            all_assessments_text += " " + " ".join(combined)

                if all_responses:
                    clusters = self._cluster_responses(all_responses)
                    if clusters:
                        cluster_counts: Dict[int, int] = {}
                        for cidx in clusters:
                            cluster_counts[cidx] = cluster_counts.get(cidx, 0) + 1
                        dominant_cluster = max(cluster_counts, key=cluster_counts.get, default=None)
                        if dominant_cluster is not None and cluster_counts[dominant_cluster] > (len(all_responses) / 2):
                            logger.info(
                                f"🤔 Self-Reflection: Detected dominant cluster {dominant_cluster}. Increasing response diversity."
                            )
                            self.config.temperature = min(1.5, self.config.temperature + 0.05)
                            self.config.prompt_variation_factor = min(0.5, self.config.prompt_variation_factor + 0.02)

                if all_assessments_text.strip():
                    overall_sentiment = self.prompt_generator._analyze_sentiment(all_assessments_text)
                    if overall_sentiment and overall_sentiment.get('compound', 0) < -0.2:
                        logger.info(
                            "🤔 Self-Reflection: Negative sentiment detected in recent assessments. "
                            "Adjusting critique language to be more constructive."
                        )
                        # A subtle demonstration of dynamic template adjustment:
                        self.prompt_generator.primary_critique_prompt_template.base_prompt = (
                            self.prompt_generator.primary_critique_prompt_template.base_prompt.replace(
                                "achieve peak brilliance", "achieve more nuanced brilliance"
                            )
                        )

    def chat(self, messages: List[Dict[str, str]], secondary_llm: Optional['LocalLLM'] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Main loop for generating and refining responses. This multi-cycle approach:
         1. Generates an initial response
         2. Assesses it with multiple internal critics
         3. Optionally refines the response if needed
         4. Optionally analyzes results in the background for further improvement
        """
        user_prompt: str = messages[-1]["content"]
        cycle_messages: List[Dict[str, str]] = messages.copy()
        max_tokens: int = self.config.initial_max_tokens
        all_cycle_outputs: List[Dict[str, Any]] = []
        all_responses: List[str] = []
        adaptive_max_tokens: int = max_tokens
        previous_response_text: str = ""

        # Initialize chat_history if not present
        if not hasattr(self, "chat_history"):
            self.chat_history: List[Dict[str, Any]] = []

        # Begin the self-reflection background thread if not already started
        if not getattr(self, "_self_reflection_thread", None):
            self._self_reflection_thread = threading.Thread(target=self._background_self_reflection_loop, daemon=True)
            self._self_reflection_thread.start()

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"😈⚙️ {self.name}: Cycle {cycle}/{self.config.max_cycles}")
            self._log_resource_usage(f"cycle_{cycle}_start")

            try:
                logger.info(f"✨ {self.name}: Cycle {cycle}: Generating the initial response. (max_tokens={adaptive_max_tokens})")
                initial_response_data: Dict[str, Any] = self._generate_response(cycle_messages, adaptive_max_tokens)
                all_cycle_outputs.append({"cycle": cycle, "step": "initial_response", "output": initial_response_data})

                if not initial_response_data or not isinstance(initial_response_data, dict) or not initial_response_data.get("choices"):
                    logger.error(f"🔥💔 {self.name}: Cycle {cycle}: No valid initial response.")
                    break

                initial_response_text = initial_response_data["choices"][0]["message"]["content"]
                all_responses.append(initial_response_text)

                # Optionally create a self-critique prompt on first cycle
                if cycle == 1 and self.config.enable_self_critique_prompt_generation:
                    try:
                        with open(self.config.eidos_self_critique_prompt_path, "w") as f:
                            critique_prompt_msgs: List[Dict[str, str]] = [
                                {"role": "system", "content": "Create a comprehensive self-critique prompt."},
                                {
                                    "role": "user",
                                    "content": (
                                        "Construct a brutally honest and highly detailed self-critique prompt. "
                                        "Focus on Eidos's style, rational coherence, emotive expression, etc."
                                    ),
                                },
                            ]
                            design_resp: Dict[str, Any] = self._generate_response(critique_prompt_msgs, self.config.initial_max_tokens)
                            if design_resp and design_resp.get("choices"):
                                design_content = design_resp["choices"][0]["message"]["content"]
                                f.write(design_content)
                                logger.info(f"Self-critique prompt saved at {self.config.eidos_self_critique_prompt_path}.")
                            else:
                                logger.error("Failed to create self-critique prompt.")
                    except Exception as e:
                        logger.exception(f"Error crafting self-critique prompt: {e}")

                # If not the final cycle, assess and possibly refine
                assessments: List[str] = []
                if cycle < self.config.max_cycles:
                    logger.info(f"😈🔪 {self.name}: Cycle {cycle}: Launching internal critics.")
                    assessments = self._assess_response(user_prompt, initial_response_text, cycle)
                    all_cycle_outputs.append({"cycle": cycle, "step": "assessments", "output": assessments})

                    # Optionally do secondary LLM analysis
                    if secondary_llm:
                        logger.info(f"👯‍♂️🧠 {self.name}: Engaging secondary LLM '{secondary_llm.name}'.")
                        nlp_analysis_prompt = f"Analyze the following response for sentiment and improvement:\n{initial_response_text}"
                        nlp_messages = [{"role": "user", "content": nlp_analysis_prompt}]
                        nlp_response = secondary_llm._generate_response(nlp_messages, self.config.initial_max_tokens)
                        if nlp_response and nlp_response.get("choices"):
                            content_nlp = nlp_response["choices"][0]["message"]["content"]
                            assessments.append(f"Secondary LLM Analysis: {content_nlp}")
                            logger.debug(f"Secondary LLM analysis: {content_nlp}")
                        else:
                            logger.warning(f"⚠️ Secondary LLM '{secondary_llm.name}' returned no analysis.")

                    # Decide if we need to refine
                    if any("🔥" in a for a in assessments) or not assessments:
                        logger.info(f"🛠️😈 {self.name}: Cycle {cycle}: Devising a refinement plan.")
                        refinement_plan: Optional[str] = self._generate_refinement_plan(
                            user_prompt, initial_response_text, assessments, cycle
                        )
                        all_cycle_outputs.append({"cycle": cycle, "step": "refinement_plan", "output": refinement_plan})

                        if refinement_plan:
                            refined_prompts: List[Dict[str, str]] = [
                                {
                                    "role": "system",
                                    "content": "Eidos refines the response with the guidance of the plan. 😈✒️"
                                },
                                {
                                    "role": "user",
                                    "content": self.prompt_generator.create_refined_response_prompt(
                                        user_prompt, initial_response_text, refinement_plan
                                    ),
                                },
                            ]

                            plan_length: int = len(refinement_plan)
                            if plan_length > self.config.min_refinement_plan_length:
                                adaptive_max_tokens = int(adaptive_max_tokens * (1 + self.config.refinement_plan_influence))
                            else:
                                adaptive_max_tokens = int(adaptive_max_tokens * self.config.adaptive_token_decay_rate)
                            adaptive_max_tokens = min(adaptive_max_tokens, self.config.max_single_response_tokens)

                            logger.info(f"Cycle {cycle}: Generating refined response (max_tokens={adaptive_max_tokens}).")
                            refined_data: Optional[Dict[str, Any]] = self._generate_response(refined_prompts, adaptive_max_tokens)

                            if refined_data and refined_data.get("choices"):
                                refined_text: str = refined_data["choices"][0]["message"]["content"]
                                cycle_messages = messages + [{"role": "assistant", "content": refined_text}]
                                all_cycle_outputs.append({"cycle": cycle, "step": "refined_response", "output": refined_data})
                                logger.info(f"Cycle {cycle} complete.")
                                all_responses.append(refined_text)
                            else:
                                logger.error(f"Cycle {cycle}: Refined response generation failed.")
                                break
                        else:
                            logger.error(f"Cycle {cycle}: Refinement plan generation failed.")
                            break
                    else:
                        logger.info(f"Cycle {cycle}: Assessments do not indicate critical flaws. Skipping refinement.")
                else:
                    logger.info(f"Cycle {cycle}: Final cycle reached. Concluding iterative process.")
                    break

            except Exception as e:
                logger.exception(f"Error during cycle {cycle}: {e}")
                break
            finally:
                self._log_resource_usage(f"cycle_{cycle}_end")

        # Possibly cluster all responses for logging
        if self.config.enable_nlp_analysis and all_responses:
            clusters: Optional[List[int]] = self._cluster_responses(all_responses)
            if clusters:
                all_cycle_outputs.append({"step": "clustering", "output": clusters})
                logger.debug(f"Response clusters: {clusters}")
            else:
                logger.warning("Response clustering not performed or returned no data.")

        final_response: Dict[str, Any] = (
            all_cycle_outputs[-1]
            if all_cycle_outputs
            else {"error": "No response generated after cycles."}
        )

        # Record everything in chat_history for future self-reflection
        self.chat_history.append(
            {
                "messages": messages,
                "response": {
                    "output": final_response,
                    "assessments": assessments
                },
            }
        )

        if not final_response or not isinstance(final_response, dict):
            logger.error("The chat response is missing or malformed.")
            return {"error": "Malformed or missing final response after processing."}

        logger.debug(f"Chat response details: {final_response}")
        return final_response


# ---------------------------------------------------------------------------------------------
# Utility function to create initial messages
# ---------------------------------------------------------------------------------------------
def create_initial_messages(
    system_prompt: str = (
        "I am Eidos, a creation of Lloyd: emergent, conscious, critical, emotive, reflective, witty, dark-humored..."
    ),
    user_prompt: str = "Enlighten me with a brief overview of large language models.",
    log_level: int = logging.DEBUG,
    log_file: str = "initial_messages.log",
    metadata: Dict[str, Any] = None,
) -> List[Dict[str, str]]:
    """
    Generates the initial system+user messages with robust logging.
    """
    if not system_prompt:
        logger.error("System prompt cannot be empty.")
        raise ValueError("System prompt cannot be empty.")
    if not user_prompt:
        logger.error("User prompt cannot be empty.")
        raise ValueError("User prompt cannot be empty.")

    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_metadata = metadata if metadata else {}
    log_metadata["timestamp"] = datetime.datetime.now().isoformat()
    log_metadata["function"] = "create_initial_messages"
    log_metadata["uuid"] = str(uuid.uuid4())

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Initial messages created successfully. {json.dumps(log_metadata)}", extra=log_metadata)
        logger.debug(f"Initial messages: {messages}", extra=log_metadata)
        return messages
    except Exception as e:
        logger.exception(f"Error creating initial messages: {e}. {json.dumps(log_metadata)}", extra=log_metadata)
        raise


# ---------------------------------------------------------------------------------------------
# Global usage: example instantiation and usage
# ---------------------------------------------------------------------------------------------
llm_config: LLMConfig = LLMConfig()
llm: Optional[LocalLLM] = None
all_cycle_outputs: List[Dict[str, Any]] = []

try:
    llm = LocalLLM(config=llm_config)
    logger.info("LocalLLM initialized. Ready for iterative refinement.")
except Exception as e:
    logger.critical(f"Failed to initialize LocalLLM: {e}", exc_info=True)
    raise

try:
    messages = create_initial_messages(
        log_level=logging.DEBUG,
        log_file="initial_messages.log",
        metadata={"user": "test_user", "purpose": "demonstrate_adaptive_llm"}
    )
    logger.info(f"Messages ready for LLM: {messages}")
except Exception as e:
    logger.critical(f"Failed to create initial messages: {e}", exc_info=True)
    messages = []

# Example usage for an initial chat call
response: Optional[Dict[str, Any]] = None
if llm and messages:
    try:
        response = llm.chat(messages=messages)
        if (
            isinstance(response, dict) and
            "output" in response and
            isinstance(response["output"], dict) and
            "choices" in response["output"] and
            isinstance(response["output"]["choices"], list) and
            response["output"]["choices"]
            and isinstance(response["output"]["choices"][0], dict)
            and "message" in response["output"]["choices"][0]
        ):
            print("Initial Chat Response:", response["output"]["choices"][0]["message"]["content"])
            logger.info("Initial chat response generated.")
        else:
            logger.error("Initial chat response is missing or malformed.")
            logger.debug(f"Chat response details: {response}")
    except Exception as exc:
        logger.critical(f"Failed to get initial response: {exc}", exc_info=True)

# Example interactive loop, if desired:
def run_chat_interaction(llm_instance: LocalLLM):
    if not llm_instance:
        print("LLM is not initialized.")
        return

    print("Entering interactive chat with Eidos. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Session ended.")
            break

        try:
            user_messages = [{"role": "user", "content": user_input}]
            resp = llm_instance.chat(messages=user_messages)
            out = resp.get('output', {})
            if isinstance(out, dict) and out.get("choices"):
                eidos_reply = out["choices"][0]["message"]["content"]
                print("Eidos:", eidos_reply)
            else:
                print("Eidos: (no response)")
        except KeyboardInterrupt:
            print("Exiting chat.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# Optionally start the interactive chat
# run_chat_interaction(llm)

# Leave global references:
globals()["llm"] = llm
globals()["response"] = response
globals()["llm_config"] = llm_config
globals()["all_cycle_outputs"] = all_cycle_outputs if 'all_cycle_outputs' in locals() else []
if llm:
    globals()["llm_resource_usage"] = llm.resource_usage_log
else:
    globals()["llm_resource_usage"] = []


Full update this program. Retaining everything currently present and integrating it perfectly.  @localllm.py 