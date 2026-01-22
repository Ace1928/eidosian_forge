from reportlab.rl_config import register_reset
def format_french(num):
    return ('un', 'deux', 'trois', 'quatre', 'cinq')[(num - 1) % 5]