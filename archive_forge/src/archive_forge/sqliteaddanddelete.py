import sqlite3

# Connect to the database (creates a new database if it doesn't exist)
conn = sqlite3.connect("example.db")

# Create a table
conn.execute(
    """CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY,
             name TEXT,
             email TEXT)"""
)

# Insert data into the table
conn.execute(
    "INSERT INTO users (name, email) VALUES (?, ?)", ("John Doe", "john@example.com")
)
conn.execute(
    "INSERT INTO users (name, email) VALUES (?, ?)", ("Jane Smith", "jane@example.com")
)

# Commit the changes
conn.commit()

# Query data from the table
cursor = conn.execute("SELECT * FROM users")
for row in cursor:
    print(f"ID: {row[0]}, Name: {row[1]}, Email: {row[2]}")

# Close the connection
conn.close()
