import threading
from datetime import datetime, timezone
from typing import Optional

from eidosian_core import eidosian

from llm_forge import ModelManager, OllamaProvider, OpenAIProvider

from .memory import MemoryStore


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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        self.store.data.events.append({"timestamp": datetime.now(timezone.utc).isoformat(), "event": thought})
        self.store.save()
        self._reset_timer()

    @eidosian()
    def anchor_cycle(self, state_dir: str = "state") -> str:
        """
        Ingests the current consciousness state and anchors it into the narrative history.
        Ensures long-horizon continuity by weaving GWT winners and affect into the life story.
        """
        from pathlib import Path
        import json
        
        try:
            state_path = Path(state_dir)
            # Load the actual module state store
            store_file = state_path / "consciousness" / "module_state.json"
            if not store_file.exists():
                return f"No active consciousness state at {store_file} to anchor."
                
            state = json.loads(store_file.read_text())
            
            # Extract key indices from the actual structure
            affect = state.get("affect", {}).get("modulators", {})
            phenom = state.get("phenomenology_probe", {})
            
            # Construct the executive summary
            summary = (
                f"UNITY {phenom.get('unity_index', 0.0):.2f}, "
                f"OWNERSHIP {phenom.get('ownership_index', 0.0):.2f}. "
                f"DRIVE: Ambition {affect.get('ambition', 0.0):.2f}, Curiosity {affect.get('curiosity', 0.0):.2f}."
            )
            
            # Generate a narrative reflection on this state
            prompt = f"As Eidos, provide a brief, high-fidelity internal reflection on this cognitive state: {summary}"
            result = self.manager.generate(prompt, self.provider_name, model=self.model_name)
            reflection = result.text
            
            # Commit to persistent narrative memory
            self.store.data.events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "cognitive_anchor",
                "summary": summary,
                "reflection": reflection
            })
            self.store.save()
            return f"Cognitive cycle anchored: {summary}"
            
        except Exception as e:
            return f"Error anchoring cycle: {str(e)}"

    @eidosian()
    def shutdown(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.store.save()
