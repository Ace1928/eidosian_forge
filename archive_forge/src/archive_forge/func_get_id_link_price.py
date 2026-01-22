from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def get_id_link_price(self, games: dict) -> dict:
    """The response may contain more than one game, so we need to choose the right
        one and return the id."""
    game_info = {}
    for app in games['apps']:
        game_info['id'] = app['id']
        game_info['link'] = app['link']
        game_info['price'] = app['price']
        break
    return game_info