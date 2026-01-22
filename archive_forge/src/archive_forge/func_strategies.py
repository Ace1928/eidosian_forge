from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any, Tuple, Union
@jwt_required()
@api_bp.route('/strategies', methods=['GET', 'POST'])
def strategies() -> APIResponse:
    """
    Endpoint to fetch or update trading strategies. Demonstrates handling of GET and POST requests with JWT authentication.
    Returns:
        APIResponse: A tuple containing a Flask Response object with a JSON message and an appropriate status code.
    """
    if request.method == 'GET':
        return (jsonify({'strategies': list(strategies_db.values())}), 200)
    elif request.method == 'POST':
        strategy_data: StrategyData = request.get_json()
        strategy_name: str = strategy_data.get('name', '')
        if not strategy_name:
            return (jsonify({'msg': 'Strategy name is required'}), 400)
        strategies_db[strategy_name] = strategy_data
        return (jsonify({'msg': f'Strategy {strategy_name} added successfully'}), 201)