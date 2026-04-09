from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import json
import datetime
from network_anomaly_detector import (
    load_and_preprocess, load_saved_model, is_model_trained, COLUMNS,
    get_attack_category, get_severity, compute_source_behavior,
    compute_fp_fn_explanation, build_incident_conclusion, MITRE_MAPPING
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

alerts_log = []


@app.route('/')
def index():
    trained = is_model_trained('models/')
    return render_template('index.html', model_trained=trained)


@app.route('/api/status')
def status():
    return jsonify({'model_trained': is_model_trained('models/')})


@app.route('/api/predict', methods=['POST'])
def predict():
    global alerts_log

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not is_model_trained('models/'):
        return jsonify({'error': 'Model not trained yet. Run train.py first.'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.txt')
    f.save(filepath)

    try:
        model, encoders, scaler = load_saved_model('models/')
        X, y, _, _ = load_and_preprocess(filepath, encoders=encoders, scaler=scaler, fit=False)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        raw_df = pd.read_csv(filepath, header=None, names=COLUMNS)
        raw_df.drop('difficulty', axis=1, inplace=True)

        results = []
        new_alerts = []
        category_counts = {}

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            row = raw_df.iloc[i]
            confidence = float(max(prob)) * 100
            label_str = 'Attack' if pred == 1 else 'Normal'
            actual_label = str(row['label']).strip()
            actual_bin = 0 if actual_label == 'normal' else 1

            # Attack category + severity
            raw_label = actual_label if pred == 1 else 'normal'
            attack_cat = get_attack_category(raw_label)
            severity = get_severity(attack_cat)

            if attack_cat not in ('Normal',):
                category_counts[attack_cat] = category_counts.get(attack_cat, 0) + 1

            entry = {
                'index': i + 1,
                'protocol': str(row['protocol_type']),
                'service': str(row['service']),
                'flag': str(row['flag']),
                'src_bytes': int(row['src_bytes']),
                'dst_bytes': int(row['dst_bytes']),
                'prediction': label_str,
                'confidence': round(confidence, 1),
                'actual': 'Attack' if actual_bin == 1 else 'Normal',
                'correct': bool(pred == actual_bin),
                'attack_category': attack_cat,
                'severity': severity['level'],
            }
            results.append(entry)

            if pred == 1:
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                alert = {
                    'time': ts,
                    'sample': i + 1,
                    'protocol': str(row['protocol_type']),
                    'service': str(row['service']),
                    'confidence': round(confidence, 1),
                    'attack_category': attack_cat,
                    'severity': severity['level'],
                }
                new_alerts.append(alert)

        alerts_log = new_alerts[:200] + alerts_log
        alerts_log = alerts_log[:500]

        total = len(predictions)
        attacks = int(sum(predictions))
        normal = total - attacks
        correct = sum(1 for r in results if r['correct'])
        accuracy = round(correct / total * 100, 2) if total > 0 else 0

        tp = sum(1 for r in results if r['prediction'] == 'Attack' and r['actual'] == 'Attack')
        tn = sum(1 for r in results if r['prediction'] == 'Normal' and r['actual'] == 'Normal')
        fp = sum(1 for r in results if r['prediction'] == 'Attack' and r['actual'] == 'Normal')
        fn = sum(1 for r in results if r['prediction'] == 'Normal' and r['actual'] == 'Attack')

        precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0
        recall = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0
        f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0

        summary = {
            'total': total, 'attacks': attacks, 'normal': normal,
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        }

        # Source behavior analysis
        behavior = compute_source_behavior(raw_df, predictions)

        # FP/FN explanation
        fp_fn = compute_fp_fn_explanation(tp, tn, fp, fn)

        # MITRE mapping for detected categories
        detected_cats = list(set(r['attack_category'] for r in results if r['attack_category'] not in ('Normal', 'NONE')))
        mitre_data = {}
        for cat in detected_cats:
            mitre_data[cat] = MITRE_MAPPING.get(cat, MITRE_MAPPING['Unknown'])

        # Incident conclusion
        conclusion = build_incident_conclusion(summary, category_counts, behavior, fp_fn)

        return jsonify({
            'results': results[:200],
            'total_results': total,
            'summary': summary,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'alerts': new_alerts[:50],
            'category_counts': category_counts,
            'mitre': mitre_data,
            'behavior': behavior,
            'fp_fn': fp_fn,
            'conclusion': conclusion,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/alerts')
def get_alerts():
    return jsonify({'alerts': alerts_log[:100]})


@app.route('/api/download')
def download():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.txt')
    if not os.path.exists(filepath):
        return jsonify({'error': 'No predictions to download'}), 400
    if not is_model_trained('models/'):
        return jsonify({'error': 'Model not trained'}), 400

    model, encoders, scaler = load_saved_model('models/')
    X, y, _, _ = load_and_preprocess(filepath, encoders=encoders, scaler=scaler, fit=False)
    predictions = model.predict(X)
    probs = model.predict_proba(X)

    raw_df = pd.read_csv(filepath, header=None, names=COLUMNS)
    raw_df.drop('difficulty', axis=1, inplace=True)

    out = raw_df.copy()
    out['predicted_label'] = ['Attack' if p == 1 else 'Normal' for p in predictions]
    out['confidence_%'] = [round(float(max(p)) * 100, 2) for p in probs]

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='anomaly_predictions.csv'
    )


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  ANOMALYX — SOC INTELLIGENCE DASHBOARD v2.0")
    if is_model_trained('models/'):
        print("  Model loaded and ready")
    else:
        print("  Model not found. Run train.py first.")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
