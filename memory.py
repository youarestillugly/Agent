"""SQLite episodic memory and user feedback storage."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

class AgentMemory:
    def __init__(self, db_path="outputs/agent_memory.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, goal TEXT, result_json TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, rating INTEGER,
                comment TEXT, issue_category TEXT)""")

    def save_episode(self, goal, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO episodes(created_at, goal, result_json) VALUES(?,?,?)",
                         (datetime.now(timezone.utc).isoformat(), goal, json.dumps(result, default=str)))

    def recent_episodes(self, limit=5):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT created_at, goal FROM episodes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"created_at": r[0], "goal": r[1]} for r in rows]

    def save_feedback(self, rating, comment, issue_category):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO feedback(created_at, rating, comment, issue_category) VALUES(?,?,?,?)",
                         (datetime.now(timezone.utc).isoformat(), int(rating), comment, issue_category))

    def feedback_summary(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT rating, issue_category FROM feedback ORDER BY id DESC LIMIT 20").fetchall()
        if not rows:
            return ""
        average = sum(r[0] for r in rows) / len(rows)
        categories = {}
        for _, category in rows:
            categories[category] = categories.get(category, 0) + 1
        return {"feedback_count": len(rows), "average_rating": round(average, 2),
                "frequent_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)}
