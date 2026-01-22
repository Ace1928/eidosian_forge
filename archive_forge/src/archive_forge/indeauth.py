"""
Module: bp_auth.py

This module handles user authentication and registration endpoints for the trading bot application.
It uses Flask Blueprint to organize the authentication-related routes and functions.

Dependencies:
- Flask: A web framework for building the API endpoints.
- Werkzeug: A comprehensive WSGI web application library, providing security utilities.
- Flask-JWT-Extended: A Flask extension for handling JSON Web Tokens (JWTs) for authentication.
- typing: Provides type hinting support for better code readability and maintainability.

Classes:
- None

Functions:
- register_user(user_data: UserData) -> ResponseData:
    Registers a new user by hashing their password and storing user data in a simulated database.
- register() -> Tuple[ResponseData, int]:
    Endpoint for user registration.
- authenticate_user(username: str, password: str) -> ResponseData:
    Authenticates a user by checking their password against the hashed version stored in the database.
- login() -> Tuple[ResponseData, int]:
    Endpoint for user login.

Variables:
- auth_bp: Flask Blueprint instance for organizing authentication-related routes.
- users_db: Dict[str, str] - Simulated database for storing user information.
- UserData: Type alias for Dict[str, Any] representing user data.
- ResponseData: Type alias for Dict[str, Any] representing response data.
- AuthResponse: Type alias for Tuple[ResponseData, int] representing authentication response.

Authorship and Versioning Details:
        Author: Lloyd Handyside
        Creation Date: 2024-04-16 (ISO 8601 Format)
        Last Modified: 2024-04-16 (ISO 8601 Format)
        Version: 1.0.0 (Semantic Versioning)
        Contact: lloyd.handyside@neuroforge.io
        Ownership: Neuro Forge
        Status: Draft (Subject to change)
"""

from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
from typing import Dict, Any, Tuple

# Assuming an external database or a simple in-memory structure for demonstration
users_db: Dict[str, str] = {}

auth_bp = Blueprint("auth", __name__)

UserData = Dict[str, Any]
ResponseData = Dict[str, Any]
AuthResponse = Tuple[ResponseData, int]


def register_user(user_data: UserData) -> ResponseData:
    """
    Registers a new user by hashing their password and storing user data in a simulated database.

    Parameters:
    - user_data: UserData - A dictionary containing the username and password of the user.

    Returns:
    - ResponseData: A dictionary with a message indicating successful registration or an error.
    """
    username: str = user_data.get("username", "")
    password: str = user_data.get("password", "")

    if not username or not password:
        return {"msg": "Username and password are required."}

    if username in users_db:
        return {"msg": "Username already exists."}

    hashed_password: str = generate_password_hash(password)
    users_db[username] = hashed_password

    return {"msg": f"User {username} registered successfully."}


@auth_bp.route("/register", methods=["POST"])
def register() -> AuthResponse:
    """
    Endpoint for user registration.

    Returns:
    - AuthResponse: A tuple containing a Flask Response object with a JSON message and an appropriate status code.
    """
    try:
        user_data: UserData = request.get_json()
        result: ResponseData = register_user(user_data)
        status_code: int = (
            201 if "registered successfully" in result.get("msg", "") else 400
        )
        return jsonify(result), status_code
    except Exception as e:
        error_message: str = f"An error occurred during registration: {str(e)}"
        return jsonify({"msg": error_message}), 500


def authenticate_user(username: str, password: str) -> ResponseData:
    """
    Authenticates a user by checking their password against the hashed version stored in the database.

    Parameters:
    - username: str - The username of the user.
    - password: str - The password provided by the user for authentication.

    Returns:
    - ResponseData: A dictionary containing the JWT access token for the authenticated user or an error message.
    """
    hashed_password: str = users_db.get(username, "")

    if hashed_password and check_password_hash(hashed_password, password):
        access_token: str = create_access_token(identity=username)
        return {"access_token": access_token}

    return {"msg": "Invalid username or password."}


@auth_bp.route("/login", methods=["POST"])
def login() -> AuthResponse:
    """
    Endpoint for user login.

    Returns:
    - AuthResponse: A tuple containing a Flask Response object with a JSON message containing the JWT access token or an error message, and an appropriate status code.
    """
    try:
        login_data: UserData = request.get_json()
        username: str = login_data.get("username", "")
        password: str = login_data.get("password", "")
        result: ResponseData = authenticate_user(username, password)
        status_code: int = 200 if "access_token" in result else 401
        return jsonify(result), status_code
    except Exception as e:
        error_message: str = f"An error occurred during login: {str(e)}"
        return jsonify({"msg": error_message}), 500


"""
TODO:
- Integrate with a real database for storing user information securely.
- Implement additional security measures, such as rate limiting and IP blocking for failed login attempts.
- Add user roles and permissions for fine-grained access control.
- Implement user profile management endpoints for updating user information.
- Integrate with third-party authentication providers, such as OAuth or OpenID Connect.
"""
