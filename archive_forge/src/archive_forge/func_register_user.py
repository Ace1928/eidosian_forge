from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
from typing import Dict, Any, Tuple
def register_user(user_data: UserData) -> ResponseData:
    """
    Registers a new user by hashing their password and storing user data in a simulated database.

    Parameters:
    - user_data: UserData - A dictionary containing the username and password of the user.

    Returns:
    - ResponseData: A dictionary with a message indicating successful registration or an error.
    """
    username: str = user_data.get('username', '')
    password: str = user_data.get('password', '')
    if not username or not password:
        return {'msg': 'Username and password are required.'}
    if username in users_db:
        return {'msg': 'Username already exists.'}
    hashed_password: str = generate_password_hash(password)
    users_db[username] = hashed_password
    return {'msg': f'User {username} registered successfully.'}