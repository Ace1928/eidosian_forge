from flask import Flask
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
import os
from typing import Type
from scripts.trading_bot.indeplugins import jwt, socketio
from scripts.trading_bot.indeauth import auth_bp
from scripts.trading_bot.indeapi import api_bp
from scripts.trading_bot.indeconfig import Config
def create_app(config_class: Type[Config]) -> Flask:
    """
    Application factory that initializes and configures the Flask application.

    Parameters:
    - config_class: Type[Config] - The configuration class to use for application settings.

    Returns:
    - Flask: The initialized Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    jwt.init_app(app)
    socketio.init_app(app, cors_allowed_origins='*')
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(api_bp, url_prefix='/api')
    return app