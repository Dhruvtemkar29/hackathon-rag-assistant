import sqlite3

DB_NAME = "chat_history.db"

def save_message(session_id, role, message):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history
        (session_id, role, message)
        VALUES (?, ?, ?)
    """, (session_id, role, message))

    conn.commit()
    conn.close()
    
def get_chat_history(session_id):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, message
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id
    """, (session_id,))

    rows = cursor.fetchall()

    conn.close()

    return rows