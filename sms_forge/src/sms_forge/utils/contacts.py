from typing import Dict, Optional
import json
from pathlib import Path

class ContactManager:
    """Manages name-to-number mappings for the SMS Forge."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".config/eidosian/contacts.json"
        self.contacts: Dict[str, str] = {}
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                self.contacts = json.loads(self.config_path.read_text())
            except json.JSONDecodeError:
                self.contacts = {}

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.contacts, indent=2))

    def add_contact(self, name: str, number: str):
        self.contacts[name.lower()] = number
        self.save()

    def get_number(self, name: str) -> Optional[str]:
        return self.contacts.get(name.lower())

    def get_name(self, number: str) -> str:
        for name, num in self.contacts.items():
            if num == number:
                return name.capitalize()
        return number
