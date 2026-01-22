from __future__ import annotations
import typing
def normalized_messages(self):
    if self.field_name == SCHEMA and isinstance(self.messages, dict):
        return self.messages
    return {self.field_name: self.messages}