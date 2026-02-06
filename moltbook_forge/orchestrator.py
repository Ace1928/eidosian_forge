#!/usr/bin/env python3
"""
Moltbook Mission Orchestrator.
Handles task delegation and inter-agent communication protocols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class Mission:
    id: str
    target_agent: str
    objective: str
    status: str = "PENDING"  # PENDING, ACTIVE, COMPLETED, FAILED
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class MissionManager:
    """Orchestrates missions involving external Moltbook agents."""

    def __init__(self):
        self.missions: Dict[str, Mission] = {}

    def create_mission(self, target: str, objective: str) -> str:
        mission_id = f"MSN-{datetime.now().strftime('%m%d%H%M')}-{target[:4]}"
        self.missions[mission_id] = Mission(id=mission_id, target_agent=target, objective=objective)
        return mission_id

    def update_mission(self, mission_id: str, status: str, log: Optional[str] = None):
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            mission.status = status
            if log:
                mission.logs.append(f"[{datetime.now()}] {log}")

    def get_active_missions(self) -> List[Mission]:
        return [m for m in self.missions.values() if m.status in ("PENDING", "ACTIVE")]
