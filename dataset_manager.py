import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path
import sqlite3
from datetime import datetime


class SocialMediaDatasetManager:
    """Simplified dataset manager for demo purposes"""

    def __init__(self):
        self.datasets = {}
        self.combined_df = self.create_sample_dataset()
        self.setup_database()

    def create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for demo"""
        sample_data = {
            'text': [
                # Toxic examples
                "You're such a loser, nobody likes you",
                "I'm going to beat you up after school",
                "Everyone hates you, just disappear",
                "You should kill yourself",
                "You're so ugly and stupid",
                "Nobody wants you here",
                "You're worthless and pathetic",
                "Go die somewhere",
                "I wish you were never born",
                "You're a waste of space",

                # Non-toxic examples
                "Hope you have a great day!",
                "This is amazing content, thank you!",
                "Great job on the presentation!",
                "Looking forward to our meeting",
                "Congratulations on your achievement!",
                "Thanks for your help with this",
                "I appreciate your hard work",
                "Have a wonderful weekend",
                "You did an excellent job",
                "Let's work together on this",

                # More mixed examples
                "Your opinion is stupid and wrong",
                "Can't believe how dumb you are",
                "Thanks for sharing your thoughts",
                "I respectfully disagree with you",
                "You're an idiot for thinking that",
                "That's an interesting perspective",
                "Shut up and go away",
                "I value your input on this matter",
                "You're such a failure at everything",
                "Keep up the good work!"
            ],
            'label': [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Toxic
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Non-toxic
                1, 1, 0, 0, 1, 0, 1, 0, 1, 0  # Mixed
            ],
            'platform': ['twitter'] * 10 + ['facebook'] * 10 + ['instagram'] * 10,
            'dataset': ['sample'] * 30
        }

        return pd.DataFrame(sample_data)

    def setup_database(self):
        """Setup SQLite database"""
        Path('data').mkdir(exist_ok=True)
        conn = sqlite3.connect('data/cyberbullying_data.db')
        cursor = conn.cursor()

        # Create main messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                label INTEGER,
                platform TEXT,
                dataset TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create user analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                processed_text TEXT,
                platform TEXT,
                prediction TEXT,
                confidence REAL,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert sample data
        cursor.execute('SELECT COUNT(*) FROM messages')
        if cursor.fetchone()[0] == 0:
            for _, row in self.combined_df.iterrows():
                cursor.execute('''
                    INSERT INTO messages (text, label, platform, dataset)
                    VALUES (?, ?, ?, ?)
                ''', (row['text'], row['label'], row['platform'], row['dataset']))

        conn.commit()
        conn.close()

    def preprocess_text(self, text: str, platform: str = 'twitter') -> str:
        """Preprocess social media text"""
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_features(self, text: str) -> dict:
        """Extract linguistic features"""
        words = text.split()

        toxic_words = ['stupid', 'ugly', 'loser', 'hate', 'kill', 'die', 'worthless',
                       'pathetic', 'idiot', 'dumb', 'failure', 'shut', 'wrong']

        features = {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_threat': int(any(word in text for word in ['kill', 'die', 'beat', 'hurt'])),
            'has_insult': int(any(word in text for word in ['stupid', 'ugly', 'loser', 'idiot'])),
            'has_vulgarity': int(any(word in text for word in ['fuck', 'shit', 'damn', 'ass'])),
            'toxic_word_count': sum(1 for word in words if word in toxic_words)
        }

        return features

    def get_dataset_stats(self) -> dict:
        """Get dataset statistics"""
        total = len(self.combined_df)
        toxic_count = self.combined_df['label'].sum()

        return {
            'total_messages': total,
            'toxic_count': toxic_count,
            'non_toxic_count': total - toxic_count,
            'toxicity_rate': (toxic_count / total * 100) if total > 0 else 0,
            'platform_distribution': self.combined_df['platform'].value_counts().to_dict(),
            'dataset_distribution': self.combined_df['dataset'].value_counts().to_dict()
        }