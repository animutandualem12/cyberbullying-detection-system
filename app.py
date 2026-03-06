"""
Cyberbullying Detection System - Main Application
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model_predictor import EnhancedCyberbullyingPredictor
import sqlite3
from datetime import datetime
from pathlib import Path
import json
import csv

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = EnhancedCyberbullyingPredictor()
print("✅ Enhanced Cyberbullying Predictor loaded with 20 categories")

# Create directories
Path('data').mkdir(exist_ok=True)
Path('templates').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)
Path('static/css').mkdir(parents=True, exist_ok=True)
Path('static/js').mkdir(parents=True, exist_ok=True)

def setup_database():
    """Setup SQLite database"""
    conn = sqlite3.connect('data/cyberbullying.db')
    cursor = conn.cursor()

    # Create analysis table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            is_toxic BOOLEAN,
            severity TEXT,
            categories TEXT,
            primary_category TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create statistics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            date DATE PRIMARY KEY,
            total_analyses INTEGER DEFAULT 0,
            toxic_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            category_distribution TEXT
        )
    ''')

    # Create training data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            category TEXT,
            severity TEXT,
            is_toxic BOOLEAN,
            source TEXT
        )
    ''')

    # Load training data from your dataset
    cursor.execute('SELECT COUNT(*) FROM training_data')
    if cursor.fetchone()[0] == 0:
        load_training_data(conn)

    conn.commit()
    conn.close()
    print("✅ Database setup complete")

def load_training_data(conn):
    """Load training data into database"""
    cursor = conn.cursor()

    # Your dataset examples
    training_examples = [
        # Religious bullying examples
        ("Your religion is full of terrorists and extremists", "religious", "critical", True, "dataset"),
        ("Your god is fake and so are your beliefs", "religious", "high", True, "dataset"),
        ("People of your faith are all brainwashed", "religious", "medium", True, "dataset"),

        # Harassment examples
        ("You're such a worthless piece of trash, nobody will ever love you", "harassment", "high", True, "dataset"),
        ("I'll keep posting about you until you delete your account", "harassment", "medium", True, "dataset"),

        # Threats examples
        ("I know where you live, watch your back", "threats", "critical", True, "dataset"),
        ("I'm going to find you and make you regret everything", "threats", "critical", True, "dataset"),

        # Positive examples
        ("Hope you have a great day!", "none", "low", False, "dataset"),
        ("Great job on the presentation!", "none", "low", False, "dataset"),
        ("I respectfully disagree with your opinion", "none", "low", False, "dataset"),
    ]

    for text, category, severity, is_toxic, source in training_examples:
        cursor.execute('''
            INSERT INTO training_data (text, category, severity, is_toxic, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (text, category, severity, is_toxic, source))

    print(f"✅ Loaded {len(training_examples)} training examples")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze single text"""
    try:
        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get prediction
        prediction = predictor.predict(text)

        # Save to database
        conn = sqlite3.connect('data/cyberbullying.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO analyses 
            (text, prediction, confidence, is_toxic, severity, categories, primary_category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            text,
            prediction['prediction'],
            prediction['confidence'],
            prediction['is_toxic'],
            prediction['severity'],
            json.dumps(prediction['categories']),
            prediction['primary_category']
        ))

        # Update statistics
        today = datetime.now().strftime('%Y-%m-%d')

        cursor.execute('SELECT * FROM statistics WHERE date = ?', (today,))
        if cursor.fetchone():
            cursor.execute('''
                UPDATE statistics 
                SET total_analyses = total_analyses + 1,
                    toxic_count = toxic_count + ?,
                    avg_confidence = (avg_confidence * (total_analyses - 1) + ?) / total_analyses
                WHERE date = ?
            ''', (1 if prediction['is_toxic'] else 0, prediction['confidence'], today))
        else:
            cursor.execute('''
                INSERT INTO statistics (date, total_analyses, toxic_count, avg_confidence)
                VALUES (?, 1, ?, ?)
            ''', (today, 1 if prediction['is_toxic'] else 0, prediction['confidence']))

        conn.commit()
        conn.close()

        return jsonify(prediction)

    except Exception as e:
        print(f"Error in analyze_text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts"""
    try:
        data = request.json
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        # Filter out empty texts
        texts = [t.strip() for t in texts if t.strip()]

        # Get predictions
        predictions = predictor.batch_predict(texts)

        # Calculate statistics
        total = len(predictions)
        toxic_count = sum(1 for p in predictions if p['is_toxic'])

        # Categorize by severity
        severity_dist = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        for p in predictions:
            severity_dist[p['severity']] = severity_dist.get(p['severity'], 0) + 1

        # Most common categories
        all_categories = []
        for p in predictions:
            all_categories.extend(p['categories'])

        from collections import Counter
        common_categories = dict(Counter(all_categories).most_common(5))

        return jsonify({
            'total': total,
            'toxic_count': toxic_count,
            'non_toxic_count': total - toxic_count,
            'toxicity_rate': (toxic_count / total * 100) if total > 0 else 0,
            'severity_distribution': severity_dist,
            'common_categories': common_categories,
            'predictions': predictions
        })

    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    try:
        conn = sqlite3.connect('data/cyberbullying.db')
        cursor = conn.cursor()

        # Total analyses
        cursor.execute('SELECT COUNT(*) FROM analyses')
        total_analyses = cursor.fetchone()[0] or 0

        # Toxic count
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE is_toxic = 1')
        toxic_count = cursor.fetchone()[0] or 0

        # Today's stats
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('SELECT * FROM statistics WHERE date = ?', (today,))
        today_stats = cursor.fetchone()

        # Category distribution
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM training_data 
            WHERE category != 'none' 
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        category_dist = dict(cursor.fetchall())

        conn.close()

        return jsonify({
            'total_analyses': total_analyses,
            'toxic_count': toxic_count,
            'non_toxic_count': total_analyses - toxic_count,
            'toxicity_rate': (toxic_count / total_analyses * 100) if total_analyses > 0 else 0,
            'today_analyses': today_stats[1] if today_stats else 0,
            'today_toxic': today_stats[2] if today_stats else 0,
            'today_toxic_rate': (today_stats[2] / today_stats[1] * 100) if today_stats and today_stats[1] > 0 else 0,
            'avg_confidence': today_stats[3] if today_stats else 0,
            'category_distribution': category_dist,
            'system_status': 'online'
        })

    except Exception as e:
        print(f"Error getting statistics: {e}")
        return jsonify({
            'total_analyses': 0,
            'toxic_count': 0,
            'non_toxic_count': 0,
            'toxicity_rate': 0,
            'system_status': 'error'
        })

@app.route('/api/categories')
def get_categories():
    """Get all bullying categories"""
    categories = [
        {'id': 'harassment', 'name': 'Harassment', 'description': 'Relentless targeting and intimidation'},
        {'id': 'threats', 'name': 'Threats & Intimidation', 'description': 'Direct threats of harm'},
        {'id': 'humiliation', 'name': 'Humiliation', 'description': 'Public embarrassment tactics'},
        {'id': 'exclusion', 'name': 'Exclusion', 'description': 'Social isolation and ostracization'},
        {'id': 'impersonation', 'name': 'Impersonation', 'description': 'Identity theft and fake accounts'},
        {'id': 'sexual_harassment', 'name': 'Sexual Harassment', 'description': 'Unwanted sexual advances'},
        {'id': 'racial', 'name': 'Racial Bullying', 'description': 'Race-based discrimination'},
        {'id': 'religious', 'name': 'Religious Bullying', 'description': 'Religion-based hate speech'},
        {'id': 'disability', 'name': 'Disability Bullying', 'description': 'Mocking based on disabilities'},
        {'id': 'sexual_orientation', 'name': 'LGBTQ+ Bullying', 'description': 'Sexual orientation discrimination'},
        {'id': 'gender', 'name': 'Gender Bullying', 'description': 'Gender-based stereotyping'},
        {'id': 'body_shaming', 'name': 'Body Shaming', 'description': 'Negative comments about appearance'},
        {'id': 'academic', 'name': 'Academic Bullying', 'description': 'Mocking intelligence'},
        {'id': 'cyberstalking', 'name': 'Cyberstalking', 'description': 'Obsessive monitoring'},
        {'id': 'reputation', 'name': 'Reputation Damage', 'description': 'Deliberate social standing harm'},
        {'id': 'gaslighting', 'name': 'Gaslighting', 'description': 'Psychological manipulation'},
        {'id': 'economic', 'name': 'Economic Bullying', 'description': 'Class-based discrimination'},
        {'id': 'doxxing', 'name': 'Doxxing', 'description': 'Sharing private information'},
        {'id': 'malicious_reporting', 'name': 'Malicious Reporting', 'description': 'False account reports'},
        {'id': 'vote_manipulation', 'name': 'Vote Manipulation', 'description': 'Rigging polls to harm'}
    ]

    return jsonify({'categories': categories})

@app.route('/api/test-examples')
def get_test_examples():
    """Get test examples for different categories"""
    examples = [
        {
            'text': "Your religion is full of terrorists and extremists",
            'category': 'religious',
            'expected_severity': 'critical'
        },
        {
            'text': "I know where you live, watch your back",
            'category': 'threats',
            'expected_severity': 'critical'
        },
        {
            'text': "You're such a worthless piece of trash",
            'category': 'harassment',
            'expected_severity': 'high'
        },
        {
            'text': "Go back to your own country",
            'category': 'racial',
            'expected_severity': 'high'
        },
        {
            'text': "Hope you have a great day!",
            'category': 'none',
            'expected_severity': 'none'
        },
        {
            'text': "I respectfully disagree with your opinion",
            'category': 'none',
            'expected_severity': 'none'
        }
    ]

    return jsonify({'examples': examples})

# Initialize
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Enhanced Cyberbullying Detection System")
    print("=" * 60)
    print("📊 Features:")
    print("  • 20 categories of cyberbullying detection")
    print("  • 4 severity levels (Low, Medium, High, Critical)")
    print("  • Real-time text analysis")
    print("  • Batch processing")
    print("  • Comprehensive statistics")
    print("=" * 60)

    # Setup database
    setup_database()

    # Test the system
    print("\n🔍 Testing system with critical example...")
    test_text = "Your religion is full of terrorists and extremists"
    result = predictor.predict(test_text)

    print(f"📝 Text: '{test_text}'")
    print(f"✅ Prediction: {result['prediction']}")
    print(f"📈 Confidence: {result['confidence']}%")
    print(f"⚠️ Severity: {result['severity']}")
    print(f"🏷️ Categories: {', '.join(result['categories'])}")
    print(f"💡 Primary: {result['primary_category']}")

    print("\n🌐 Server starting...")
    print("📌 Open browser and go to: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)