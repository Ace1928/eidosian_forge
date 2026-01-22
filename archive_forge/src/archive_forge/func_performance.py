from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any, Tuple, Union
@jwt_required()
@api_bp.route('/performance', methods=['GET'])
def performance() -> APIResponse:
    """
    Endpoint to retrieve performance metrics of the trading bot. Demonstrates a simple GET request handling.
    Returns:
        APIResponse: A tuple containing a Flask Response object with a JSON message and an appropriate status code.
    """
    return (jsonify({'performance': list(performance_db.values())}), 200)