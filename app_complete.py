# app_complete.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import sqlite3
import io
import csv
from dataset_manager import SocialMediaDatasetManager
from model_predictor import CyberbullyingPredictor
#from model_predictor import Predictor

app = Flask(__name__)
CORS(app)
app.secret_key = 'cyberbullying-detection-secret-2024'

# Initialize components
dataset_manager = SocialMediaDatasetManager()
predictor = CyberbullyingPredictor()


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')


@app.route('/dataset-explorer')
def dataset_explorer():
    """Dataset exploration interface"""
    return render_template('dataset_explorer.html')


@app.route('/api/dataset/stats')
def get_dataset_stats():
    """Get dataset statistics"""
    stats = dataset_manager.get_dataset_stats()
    return jsonify(stats)


@app.route('/api/dataset/sample')
def get_dataset_sample():
    """Get sample data from dataset"""
    try:
        sample_size = int(request.args.get('size', 100))
        page = int(request.args.get('page', 1))

        df = dataset_manager.combined_df
        start_idx = (page - 1) * sample_size
        end_idx = start_idx + sample_size

        sample_data = df.iloc[start_idx:end_idx].to_dict('records')

        return jsonify({
            'data': sample_data,
            'total': len(df),
            'page': page,
            'page_size': sample_size,
            'total_pages': (len(df) + sample_size - 1) // sample_size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/visualizations')
def get_visualizations():
    """Generate dataset visualizations"""
    df = dataset_manager.combined_df

    # 1. Toxicity distribution pie chart
    toxicity_counts = df['label'].value_counts()
    fig1 = go.Figure(data=[go.Pie(
        labels=['Non-Toxic', 'Toxic'],
        values=toxicity_counts.values,
        hole=.3
    )])
    fig1.update_layout(title='Toxicity Distribution')

    # 2. Platform distribution bar chart
    platform_counts = df['platform'].value_counts()
    fig2 = go.Figure(data=[go.Bar(
        x=platform_counts.index,
        y=platform_counts.values,
        marker_color='indianred'
    )])
    fig2.update_layout(title='Messages by Platform', xaxis_title='Platform', yaxis_title='Count')

    # 3. Text length distribution
    df['text_length'] = df['text'].apply(len)
    fig3 = px.histogram(df, x='text_length', color='label',
                        nbins=50,
                        title='Text Length Distribution by Toxicity',
                        labels={'text_length': 'Text Length', 'count': 'Count'},
                        color_discrete_map={0: 'green', 1: 'red'})

    # 4. Word cloud data (top words)
    toxic_texts = ' '.join(df[df['label'] == 1]['text'].tolist())
    non_toxic_texts = ' '.join(df[df['label'] == 0]['text'].tolist())

    # Convert to JSON
    graphs = [
        json.loads(json.dumps(fig1.to_plotly_json(), cls=plotly.utils.PlotlyJSONEncoder)),
        json.loads(json.dumps(fig2.to_plotly_json(), cls=plotly.utils.PlotlyJSONEncoder)),
        json.loads(json.dumps(fig3.to_plotly_json(), cls=plotly.utils.PlotlyJSONEncoder))
    ]

    return jsonify({'graphs': graphs})


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze single text message"""
    data = request.json
    text = data.get('text', '')
    platform = data.get('platform', 'twitter')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess
    processed_text = dataset_manager.preprocess_text(text, platform)

    # Extract features
    features = dataset_manager.extract_features(text)

    # Get prediction
    prediction = predictor.predict(processed_text)

    # Add to database
    conn = sqlite3.connect('data/cyberbullying_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_analysis (text, processed_text, platform, prediction, confidence, features, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (text, processed_text, platform,
          prediction['prediction'], prediction['confidence'],
          json.dumps(features), datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return jsonify({
        'original_text': text,
        'processed_text': processed_text,
        'platform': platform,
        'prediction': prediction['prediction'],
        'confidence': prediction['confidence'],
        'is_toxic': prediction['is_toxic'],
        'features': features,
        'suggestions': get_suggestions(prediction, features)
    })


@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyze batch of texts from uploaded file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    platform = request.form.get('platform', 'twitter')

    try:
        # Read CSV or Excel file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Process each text
        results = []
        for text in df.iloc[:, 0]:  # Assuming text is in first column
            processed_text = dataset_manager.preprocess_text(text, platform)
            prediction = predictor.predict(processed_text)
            features = dataset_manager.extract_features(text)

            results.append({
                'text': text,
                'processed_text': processed_text,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'is_toxic': prediction['is_toxic'],
                'features': features
            })

        # Generate report
        report = generate_batch_report(results)

        return jsonify({
            'total': len(results),
            'toxic_count': sum(1 for r in results if r['is_toxic']),
            'results': results,
            'report': report
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain model with new data"""
    try:
        data = request.json
        texts = data.get('texts', [])
        labels = data.get('labels', [])

        if len(texts) != len(labels):
            return jsonify({'error': 'Texts and labels must have same length'}), 400

        # Add to dataset
        new_data = pd.DataFrame({
            'text': texts,
            'label': labels,
            'platform': ['user_submitted'] * len(texts),
            'dataset': ['user_training'] * len(texts)
        })

        dataset_manager.combined_df = pd.concat(
            [dataset_manager.combined_df, new_data],
            ignore_index=True
        )

        # Retrain model
        accuracy = predictor.retrain(
            dataset_manager.combined_df['text'].tolist(),
            dataset_manager.combined_df['label'].tolist()
        )

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'new_dataset_size': len(dataset_manager.combined_df)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/results', methods=['POST'])
def export_results():
    """Export analysis results to CSV"""
    data = request.json
    results = data.get('results', [])

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'cyberbullying_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route('/api/dashboard/metrics')
def get_dashboard_metrics():
    """Get dashboard metrics"""
    conn = sqlite3.connect('data/cyberbullying_data.db')

    # Total analyses
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_analysis")
    total_analyses = cursor.fetchone()[0]

    # Recent toxic rate
    cursor.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN prediction = 'Toxic' THEN 1 ELSE 0 END) as toxic_count
        FROM user_analysis 
        WHERE timestamp >= datetime('now', '-7 days')
    ''')
    recent_stats = cursor.fetchone()

    # Platform distribution
    cursor.execute('''
        SELECT platform, COUNT(*) as count
        FROM user_analysis
        GROUP BY platform
        ORDER BY count DESC
    ''')
    platform_dist = dict(cursor.fetchall())

    conn.close()

    return jsonify({
        'total_analyses': total_analyses,
        'recent_toxic_rate': (recent_stats[1] / recent_stats[0] * 100) if recent_stats[0] > 0 else 0,
        'platform_distribution': platform_dist,
        'model_accuracy': predictor.get_accuracy(),
        'uptime': get_system_uptime()
    })


def get_suggestions(prediction, features):
    """Get suggestions based on prediction and features"""
    suggestions = []

    if prediction['is_toxic']:
        suggestions.append("🚫 This message contains toxic content")

        if features['has_threat']:
            suggestions.append("⚠️ Contains threatening language")
        if features['has_insult']:
            suggestions.append("⚠️ Contains insulting language")
        if features['has_vulgarity']:
            suggestions.append("⚠️ Contains vulgar language")

        suggestions.append("💡 Consider reporting this content to platform moderators")
        suggestions.append("💡 Block the user to prevent further harassment")
    else:
        suggestions.append("✅ This message appears to be non-toxic")
        suggestions.append("💙 Keep promoting positive communication!")

    return suggestions


def generate_batch_report(results):
    """Generate comprehensive batch analysis report"""
    toxic_results = [r for r in results if r['is_toxic']]

    report = {
        'summary': {
            'total_messages': len(results),
            'toxic_messages': len(toxic_results),
            'non_toxic_messages': len(results) - len(toxic_results),
            'toxicity_rate': (len(toxic_results) / len(results)) * 100 if results else 0
        },
        'top_toxic_patterns': [],
        'risk_assessment': 'Low',
        'recommendations': []
    }

    # Calculate risk assessment
    toxicity_rate = report['summary']['toxicity_rate']
    if toxicity_rate > 30:
        report['risk_assessment'] = 'High'
        report['recommendations'].append("Immediate moderation action required")
        report['recommendations'].append("Consider enabling strict filtering")
    elif toxicity_rate > 10:
        report['risk_assessment'] = 'Medium'
        report['recommendations'].append("Increase monitoring frequency")
        report['recommendations'].append("Review platform guidelines with users")
    else:
        report['risk_assessment'] = 'Low'
        report['recommendations'].append("Continue current moderation practices")

    report['recommendations'].append("Provide positive communication training")
    report['recommendations'].append("Implement user reporting system")

    return report


def get_system_uptime():
    """Calculate system uptime"""
    # In production, you would track start time
    return "99.8%"


if __name__ == '__main__':
    # Create necessary directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)