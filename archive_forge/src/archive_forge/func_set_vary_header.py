def set_vary_header(response, header_name):
    """Add a Vary header to a response."""
    varies = response.headers.get('Vary', '')
    varies = [x.strip() for x in varies.split(',') if x.strip()]
    if header_name not in varies:
        varies.append(header_name)
    response.headers['Vary'] = ', '.join(varies)