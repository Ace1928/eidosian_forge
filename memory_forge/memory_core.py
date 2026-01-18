"""
Memory Forge - Episodic and semantic hybrid memory system.
Provides dual-encoding for timestamped events and general facts.
"""
from typing import Dict, Any, List, Optional, Union
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

class MemoryEntry:
    """A single record in memory."""
    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "metadata": self.metadata
        }

class EpisodicMemory:
    """Stores sequential, timestamped events."""
    def __init__(self):
        self.events: List[MemoryEntry] = []

    def record(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        entry = MemoryEntry(content, metadata)
        self.events.append(entry)
        return entry

    def get_recent(self, count: int = 10) -> List[MemoryEntry]:
        return self.events[-count:]

class SemanticMemory:
    """Stores general facts and knowledge (key-value or graph-like)."""
    def __init__(self):
        self.facts: Dict[str, MemoryEntry] = {}

    def store_fact(self, key: str, content: Any, metadata: Optional[Dict[str, Any]] = None):
        entry = MemoryEntry(content, metadata)
        self.facts[key] = entry
        return entry

    def get_fact(self, key: str) -> Optional[MemoryEntry]:
        return self.facts.get(key)

class MemoryForge:
    """
    Unified memory system combining episodic and semantic layers with persistence.
    """
    def __init__(self, persistence_path: Optional[Union[str, Path]] = None, llm: Optional['LLMForge'] = None):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.llm = llm
        
        if self.persistence_path and self.persistence_path.exists():
            self.load()

    def remember(self, content: Any, is_fact: bool = False, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Store information and persist if path is set."""
        if is_fact and key:
            entry = self.semantic.store_fact(key, content, metadata)
        else:
            entry = self.episodic.record(content, metadata)
            
        if self.persistence_path:
            self.save()
        return entry

    def retrieve(self, query: str) -> List[MemoryEntry]:
        """Simple keyword-based retrieval."""
        results = []
        with threading.Lock(): # Basic safety for simultaneous access
            for key, entry in self.semantic.facts.items():
                if query.lower() in key.lower() or query.lower() in str(entry.content).lower():
                    results.append(entry)
            for entry in self.episodic.events:
                if query.lower() in str(entry.content).lower():
                    results.append(entry)
        return results

    def consolidate(self):
        """
        Move episodic memories into semantic facts using LLM insights.
        If no LLM is provided, it performs basic metadata-based consolidation.
        """
        recent_episodes = self.episodic.get_recent(5)
        if not recent_episodes:
            return

        if self.llm:
            prompt = f"Analyze these recent events and extract 1-3 permanent facts or patterns:\n"
            for ep in recent_episodes:
                prompt += f"- {ep.timestamp}: {ep.content}\n"
            
            response = self.llm.generate(prompt, system="You are the Eidosian Memory Consolidation unit. Extract semantic facts in JSON format: {'facts': [{'key': '...', 'value': '...'}]}")
            if response["success"]:
                try:
                    # Clean the response if it contains markdown code blocks
                    text = response["response"]
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    data = json.loads(text)
                    for fact in data.get("facts", []):
                        self.remember(fact["value"], is_fact=True, key=fact["key"], metadata={"source": "consolidation"})
                except Exception:
                    pass # Fallback or log error
        else:
            # Basic consolidation: tag episodes as 'consolidated'
            for ep in recent_episodes:
                ep.metadata["consolidated"] = True
        
        if self.persistence_path:
            self.save()

    def save(self):
        """Save memory state to JSON."""
        data = {
            "episodic": [e.to_dict() for e in self.episodic.events],
            "semantic": {k: v.to_dict() for k, v in self.semantic.facts.items()}
        }
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load memory state from JSON."""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
                for item in data.get("episodic", []):
                    entry = MemoryEntry(item["content"], item["metadata"])
                    entry.id = item["id"]
                    entry.timestamp = item["timestamp"]
                    self.episodic.events.append(entry)
                for key, item in data.get("semantic", {}).items():
                    entry = MemoryEntry(item["content"], item["metadata"])
                    entry.id = item["id"]
                    entry.timestamp = item["timestamp"]
                    self.semantic.facts[key] = entry
        except Exception:
            pass
