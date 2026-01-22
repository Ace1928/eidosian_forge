from flask import Flask, request, jsonify
import requests  # This module can be used here as it's server-side
@app.route('/perform_search', methods=['POST'])
def perform_search():
    query = request.json.get('query')
    search_results = ['Result 1', 'Result 2', 'Result 3']
    return jsonify(search_results)