from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Agent:
    name: str

    def choose_move(self, board: Any, rng: Optional[random.Random] = None) -> Any:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, name: str = "random") -> None:
        super().__init__(name=name)

    def choose_move(self, board: Any, rng: Optional[random.Random] = None) -> Any:
        moves = list(board.legal_moves)
        if not moves:
            return None
        chooser = rng or random
        return chooser.choice(moves)


class AgentForgeAgent(Agent):
    def __init__(
        self,
        name: str = "agent-forge",
        policy: str = "capture",
        memory_dir: Optional[str] = None,
        git_enabled: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.policy = policy
        self._task_log: list[Any] = []

        try:
            from agent_forge.models.schemas import Task
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "agent_forge is required for the agent-forge player. "
                "Add agent_forge/src to PYTHONPATH or install the package."
            ) from exc

        self._task_cls = Task
        self._memory = None

        if memory_dir:
            try:
                from agent_forge.core.memory import MemorySystem
            except ModuleNotFoundError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "agent_forge memory system is unavailable. "
                    "Ensure agent_forge is importable."
                ) from exc
            self._memory = MemorySystem(
                memory_path=memory_dir,
                git_enabled=git_enabled,
            )

    def choose_move(self, board: Any, rng: Optional[random.Random] = None) -> Any:
        moves = list(board.legal_moves)
        if not moves:
            return None

        chooser = rng or random
        move = self._select_move(board, moves, chooser)
        self._record_task(board, move)
        return move

    def _select_move(self, board: Any, moves: list[Any], chooser: Any) -> Any:
        if self.policy == "capture":
            capture_moves = [mv for mv in moves if board.is_capture(mv)]
            if capture_moves:
                return chooser.choice(capture_moves)
        return chooser.choice(moves)

    def _record_task(self, board: Any, move: Any) -> None:
        task_id = f"agentic_chess_{uuid.uuid4().hex[:8]}"
        move_repr = self._safe_move_repr(board, move)
        description = f"{self.name} selected move {move_repr}"
        task = self._task_cls(description=description, task_id=task_id)
        task.status = "completed"
        task.result = {
            "move": move_repr,
            "fen": board.fen(),
            "policy": self.policy,
        }
        self._task_log.append(task)
        if self._memory is not None:
            self._memory.save_task(task)

    @staticmethod
    def _safe_move_repr(board: Any, move: Any) -> str:
        if move is None:
            return "none"
        try:
            return board.san(move)
        except Exception:
            try:
                return move.uci()
            except Exception:
                return str(move)
