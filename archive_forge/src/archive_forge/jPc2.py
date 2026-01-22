from ai_logic import (
    import_from_path,
    dynamic_depth_expectimax,
    adjust_depth_based_on_complexity,
    expectimax,
    heuristic_evaluation,
    simulate_move,
    get_empty_tiles,
    is_game_over,
    calculate_best_move,
    short_term_memory,
    lru_memory,
    short_to_long_term_memory_transfer,
    long_term_memory_optimisation,
    long_term_memory_learning,
    long_term_memory_pruning,
    long_term_memory_retrieval,
    long_term_memory_update,
    long_term_memory_storage,
    long_term_memory_indexing,
    long_term_memory_backup,
    long_term_memory_restore,
    long_term_memory_clear,
    long_term_memory_search,
    long_term_memory_retrieval,
    long_term_memory_update,
    short_term_memory_update,
    short_term_memory_storage,
    short_term_memory_clear,
    short_term_memory_search,
    short_term_memory_retrieval,
    short_term_memory_backup,
    short_term_memory_restore,
    short_term_memory_to_long_term_memory_transfer,
    long_term_memory_to_short_term_memory_transfer,
    optimise_game_strategy,
    learn_from_game_data,
    analyse_game_data,
    visualise_game_data,
    search_game_data,
    retrieve_game_data,
    update_game_data,
    store_game_data,
    clear_game_data,
    export_game_data,
    import_game_data,
    genetic_algorithm,
    reinforcement_learning,
    deep_q_learning,
    monte_carlo_tree_search,
    alpha_beta_pruning,
    minimax_algorithm,
    expectimax_algorithm,
    evaluate_game,
    optimise_game,
    predict_game_outcomes,
    analyse_game_state,
    search_game_state,
    retrieve_game_state,
    update_game_state,
    store_game_state,
    clear_game_state,
    export_game_state,
    import_game_state,
    analyse_prediction_accuracy,
    analyse_previous_moves,
    analyse_prediction_errors,
    anlayse_learning_progress,
    introspective_analysis,
    optimise_learning_rate,
    alter_learning_strategy,
    adapt_decision_making,
)
from gui_utils import (
    get_tile_color,
    get_tile_text,
    get_tile_font_size,
    get_tile_font_color,
    get_tile_font_weight,
    get_tile_font_family,
    update_gui,
    draw_board,
    draw_gui,
    draw_tile,
    draw_score,
    draw_game_over,
    draw_win,
    draw_game_start,
    draw_game_restart,
    draw_game_pause,
    draw_game_resume,
    draw_game_help,
    draw_game_settings,
    draw_game_exit,
    draw_game_save,
    draw_game_load,
    ui_design,
    draw_ui,
    setup_ui_events,
    ui_interaction,
)
from game_manager import (
    initialize_game,
    add_random_tile,
    update_game_state,
    efficient_game_state_update,
    check_game_over,
    process_move,
    randomise_next_tile,
    setup_event_handlers,
    on_game_start,
    on_game_end,
    on_game_restart,
    on_game_exit,
    on_game_pause,
    on_game_resume,
    on_game_win,
    on_game_loss,
    on_game_update,
    on_game_move,
    on_game_event,
    on_game_input,
    on_game_output,
)
from typing import List, Tuple
import types
import importlib.util
import logging


def import_from_path(name: str, path: str) -> types.ModuleType:
    """
    Dynamically imports a module from a given file path.

    Args:
        name (str): The name of the module.
        path (str): The file path to the module.

    Returns:
        types.ModuleType: The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


standard_decorator = import_from_path(
    "standard_decorator", "/home/lloyd/EVIE/standard_decorator.py"
)
StandardDecorator = standard_decorator.StandardDecorator
setup_logging = standard_decorator.setup_logging

setup_logging()


@StandardDecorator()
def ai_game_loop(board: np.ndarray, depth: int = 3) -> Tuple[np.ndarray, int]:
    """
    Runs the game loop using an AI agent to make moves.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The final game board state and the total score.
    """
    score = 0
    game_over = False

    while not game_over:
        best_move = calculate_best_move(board)
        board, move_score = simulate_move(board, best_move)
        score += move_score
        game_over = is_game_over(board)

        if not game_over:
            add_random_tile(board)

    return board, score


def ai_update_game_state(board: np.ndarray, depth: int = 3) -> Tuple[np.ndarray, int]:
    """
    Updates the game state using an AI agent to make moves.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the total score.
    """
    return ai_game_loop(board, depth)


def ai_game_over_check(board: np.ndarray, depth: int = 3) -> bool:
    """
    Checks if the game is over for an AI agent.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    return is_game_over(board)


def ai_process_move(board: np.ndarray, depth: int = 3) -> Tuple[np.ndarray, int]:
    """
    Processes a move on the game board using an AI agent.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the total score.
    """
    return ai_game_loop(board, depth)


@StandardDecorator()
def main_game_loop():
    board = initialize_game()
    score = 0
    game_over = False

    while not game_over:
        best_move = calculate_best_move(board)
        if best_move:
            board, move_score = simulate_move(board, best_move)
            score += move_score
            add_random_tile(board)
            update_gui(board, score)
            game_over = is_game_over(board)
        else:
            game_over = True

    print(f"Game Over! Final Score: {score}")
