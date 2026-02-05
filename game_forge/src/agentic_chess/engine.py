from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional


class DependencyError(RuntimeError):
    pass


def import_chess() -> Any:
    try:
        import chess
        import chess.pgn
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "python-chess is required. Install with: pip install python-chess"
        ) from exc
    return chess


@dataclass
class MatchConfig:
    max_moves: int = 200
    seed: Optional[int] = None
    log_moves: bool = False
    emit_pgn: bool = False


@dataclass
class MatchResult:
    result: str
    termination: str
    moves: list[str]
    pgn: Optional[str] = None


def play_match(white: Any, black: Any, config: MatchConfig) -> MatchResult:
    chess = import_chess()
    board = chess.Board()
    rng = random.Random(config.seed)
    moves_san: list[str] = []

    while True:
        if board.is_game_over():
            outcome = board.outcome()
            result = board.result()
            termination = outcome.termination.name.lower() if outcome else "game-over"
            break

        if len(moves_san) >= config.max_moves:
            result = "1/2-1/2"
            termination = "max-moves"
            break

        agent = white if board.turn == chess.WHITE else black
        move = agent.choose_move(board, rng=rng)
        move = _normalize_move(board, move, chess)
        if move is None or move not in board.legal_moves:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "illegal-move"
            break

        san = board.san(move)
        board.push(move)
        moves_san.append(san)
        if config.log_moves:
            print(f"INFO move {len(moves_san)}: {san}")

    pgn = None
    if config.emit_pgn:
        game = chess.pgn.Game.from_board(board)
        pgn = str(game)

    return MatchResult(result=result, termination=termination, moves=moves_san, pgn=pgn)


def _normalize_move(board: Any, move: Any, chess: Any) -> Any:
    if move is None:
        return None
    if isinstance(move, chess.Move):
        return move
    if isinstance(move, str):
        try:
            parsed = chess.Move.from_uci(move)
        except ValueError:
            return None
        return parsed
    return None
