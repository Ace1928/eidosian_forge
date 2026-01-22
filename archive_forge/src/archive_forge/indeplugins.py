"""
Authorship and Versioning Details:
        Author: Lloyd Handyside
        Creation Date: 2024-04-16 (ISO 8601 Format)
        Last Modified: 2024-04-16 (ISO 8601 Format)
        Version: 1.0.0 (Semantic Versioning)
        Contact: lloyd.handyside@neuroforge.io
        Ownership: Neuro Forge
        Status: Draft (Subject to change)
"""

from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO

jwt = JWTManager()
socketio = SocketIO()
