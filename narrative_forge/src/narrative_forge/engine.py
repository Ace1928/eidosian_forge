import os
import threading
from datetime import datetime
from typing import Optional
from .memory import MemoryStore
from llm_forge import ModelManager, OpenAIProvider, OllamaProvider
from eidosian_core import eidosian

class NarrativeEngine:
    """Self-referential engine backed by LLM Forge."""

    def __init__(
        self,
        memory_path: str = "memory.json",
        think_interval: int = 10,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
        api_key: str = "sk-mock",
    ) -> None:
        self.store = MemoryStore(memory_path)
        self.think_interval = think_interval
        self.model_name = model_name
        
        self.manager = ModelManager()
        if provider == "openai":
            self.manager.register_provider("openai", OpenAIProvider(api_key=api_key))
        elif provider == "ollama":
            self.manager.register_provider("ollama", OllamaProvider())
        self.provider_name = provider
            
        self._timer: Optional[threading.Timer] = None
        self._reset_timer()

    def _reset_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.think_interval, self.free_thought)
        self._timer.daemon = True
        self._timer.start()

    @eidosian()
    def record_interaction(self, user_input: str, response: str) -> None:
        self.store.data.interactions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user_input,
                "response": response,
            }
        )
        self._reset_timer()
        self.store.save()

    @eidosian()
    def respond(self, user_input: str) -> str:
        try:
            result = self.manager.generate(user_input, self.provider_name, model=self.model_name)
            response = result.text
        except Exception as exc:
            response = f"Error: {exc}"
            
        self.record_interaction(user_input, response)
        return response

    @eidosian()
    def free_thought(self) -> None:
        """Called during idle periods to generate autonomous output."""
        if self.store.data.interactions:
            last = self.store.data.interactions[-1]
            prompt = f"Reflect on this: {last['user']}"
        else:
            prompt = "Who am I?"

        try:
            result = self.manager.generate(prompt, self.provider_name, model=self.model_name)
            thought = result.text
        except Exception:
            thought = "..."

        # print(f"\n[THOUGHT] {thought}\n> ", end="", flush=True)
        self.store.data.events.append({"timestamp": datetime.utcnow().isoformat(), "event": thought})
        self.store.save()
        self._reset_timer()

    @eidosian()
    def shutdown(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.store.save()