import os 
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

load_dotenv()

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET")
        self.algorithm = "HS256"

    def generate_token(self, user_id, expires_in=3600):
        """Generate JWT token"""
        payload = {
            'sub': user_id,
            'exp': datetime.utcnow() + timedelta(second=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return None 
        except jwt.InvalidTokenError:
            return None 

    def token_required(f):
        """Decorator for protected routes"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token or not token,startswith('Bearer '):
                return jsonify({ "error": "Missing or invalid token"}), 401

            auth = AuthManager()
            user_id = auth.verify_token(token[7:])
            if not user_id:
                return jsonify({ "error": "Invalid or expires token"}), 401

            return f(user_id, *args, **kwargs)
        return decorated