
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL = ['protocol_type', 'service', 'flag']

# ── ATTACK CATEGORY MAPPING ───────────────────────────────────────────────────
ATTACK_CATEGORIES = {
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}

# ── MITRE ATT&CK MAPPING ──────────────────────────────────────────────────────
MITRE_MAPPING = {
    'DoS': [
        {'id': 'T1498', 'name': 'Network Denial of Service', 'tactic': 'Impact'},
        {'id': 'T1499', 'name': 'Endpoint Denial of Service', 'tactic': 'Impact'},
    ],
    'Probe': [
        {'id': 'T1046', 'name': 'Network Service Scanning', 'tactic': 'Discovery'},
        {'id': 'T1595', 'name': 'Active Scanning', 'tactic': 'Reconnaissance'},
    ],
    'R2L': [
        {'id': 'T1110', 'name': 'Brute Force', 'tactic': 'Credential Access'},
        {'id': 'T1078', 'name': 'Valid Accounts', 'tactic': 'Persistence'},
        {'id': 'T1133', 'name': 'External Remote Services', 'tactic': 'Initial Access'},
    ],
    'U2R': [
        {'id': 'T1068', 'name': 'Exploitation for Privilege Escalation', 'tactic': 'Privilege Escalation'},
        {'id': 'T1548', 'name': 'Abuse Elevation Control Mechanism', 'tactic': 'Privilege Escalation'},
    ],
    'Unknown': [
        {'id': 'T1059', 'name': 'Command and Scripting Interpreter', 'tactic': 'Execution'},
    ],
}

# ── SEVERITY ──────────────────────────────────────────────────────────────────
CATEGORY_SEVERITY = {
    'U2R':     {'level': 'CRITICAL', 'score': 4},
    'R2L':     {'level': 'HIGH',     'score': 3},
    'DoS':     {'level': 'HIGH',     'score': 3},
    'Probe':   {'level': 'MEDIUM',   'score': 2},
    'Unknown': {'level': 'LOW',      'score': 1},
}


def get_attack_category(label_str):
    l = str(label_str).strip().lower()
    if l == 'normal':
        return 'Normal'
    return ATTACK_CATEGORIES.get(l, 'Unknown')


def get_severity(category):
    if category == 'Normal':
        return {'level': 'NONE', 'score': 0}
    return CATEGORY_SEVERITY.get(category, CATEGORY_SEVERITY['Unknown'])


def compute_source_behavior(raw_df, predictions):
    df = raw_df.copy()
    df['_pred'] = predictions
    attack_mask = df['_pred'] == 1

    svc_counts = df[attack_mask]['service'].value_counts().head(8).to_dict()
    proto_counts = df[attack_mask]['protocol_type'].value_counts().to_dict()
    failed_logins_total = int(df['num_failed_logins'].sum())
    failed_login_attacks = int(df[attack_mask]['num_failed_logins'].sum())
    brute_force_samples = int(df[(df['num_failed_logins'] > 0) & attack_mask].shape[0])
    root_shell_count = int(df[attack_mask]['root_shell'].sum())
    su_attempted_count = int(df[attack_mask]['su_attempted'].sum())
    scan_like_count = int(df[(df['count'] > 200) & attack_mask].shape[0])
    high_serror = int(df[(df['serror_rate'] > 0.5) & attack_mask].shape[0])

    return {
        'top_services': svc_counts,
        'proto_breakdown': proto_counts,
        'failed_logins_total': failed_logins_total,
        'failed_login_attacks': failed_login_attacks,
        'brute_force_samples': brute_force_samples,
        'root_shell_count': root_shell_count,
        'su_attempted_count': su_attempted_count,
        'scan_like_count': scan_like_count,
        'high_serror_count': high_serror,
    }


def compute_fp_fn_explanation(tp, tn, fp, fn):
    fp_rate = round(fp / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0
    fn_rate = round(fn / (fn + tp) * 100, 2) if (fn + tp) > 0 else 0
    fp_note = (
        "False Positives: Normal traffic flagged as attack. "
        "Often caused by unusual-but-legitimate protocol usage, "
        "high-volume benign services (e.g. http, ftp), or edge-case flag patterns. "
        f"Current FP rate: {fp_rate}%."
    )
    fn_note = (
        "False Negatives: Attacks missed by the model. "
        "Typically novel attack variants, low-intensity probes, or "
        "R2L attacks that mimic normal authenticated sessions. "
        f"Current FN rate: {fn_rate}%."
    )
    return {'fp_note': fp_note, 'fn_note': fn_note, 'fp_rate': fp_rate, 'fn_rate': fn_rate}


def _cat_description(cat):
    return {
        'DoS':   'volumetric denial-of-service or flood attacks targeting availability',
        'Probe': 'reconnaissance or port scanning activity',
        'R2L':   'remote-to-local credential attacks or unauthorized access attempts',
        'U2R':   'local privilege escalation or rootkit deployment',
    }.get(cat, 'unclassified malicious activity')


def build_incident_conclusion(summary, category_counts, behavior, fp_fn):
    total = summary['total']
    attacks = summary['attacks']
    pct = round(attacks / total * 100, 1) if total > 0 else 0
    dominant_cat = max(category_counts, key=category_counts.get) if category_counts else 'Unknown'
    dominant_svc = list(behavior['top_services'].keys())[0] if behavior['top_services'] else 'N/A'

    lines = []
    lines.append(
        f"Analysis of {total:,} traffic samples detected {attacks:,} anomalous connections ({pct}% attack rate)."
    )
    if dominant_cat != 'Normal':
        lines.append(
            f"Dominant attack category: {dominant_cat} — indicating {_cat_description(dominant_cat)}."
        )
    if behavior['brute_force_samples'] > 0:
        lines.append(
            f"Brute-force indicators present: {behavior['brute_force_samples']} samples with failed login activity."
        )
    if behavior['root_shell_count'] > 0:
        lines.append(
            f"CRITICAL: Privilege escalation signals detected — {behavior['root_shell_count']} root shell activations."
        )
    if behavior['scan_like_count'] > 0:
        lines.append(
            f"High-rate connection bursts ({behavior['scan_like_count']} samples) suggest active scanning or DoS activity."
        )
    lines.append(
        f"Top targeted service: {dominant_svc}. "
        f"False positive rate: {fp_fn['fp_rate']}%, False negative rate: {fp_fn['fn_rate']}%."
    )
    return ' '.join(lines)


def load_and_preprocess(filepath, encoders=None, scaler=None, fit=True):
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    df.drop('difficulty', axis=1, inplace=True)
    df['label'] = df['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)

    if fit:
        encoders = {}
        for col in CATEGORICAL:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in CATEGORICAL:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    X = df.drop('label', axis=1)
    y = df['label']

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, encoders, scaler


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return y_pred, acc, report, cm


def save_model(model, encoders, scaler, path='models/'):
    joblib.dump(model, f'{path}anomaly_model.pkl')
    joblib.dump(encoders, f'{path}encoders.pkl')
    joblib.dump(scaler, f'{path}scaler.pkl')


def load_saved_model(path='models/'):
    model = joblib.load(f'{path}anomaly_model.pkl')
    encoders = joblib.load(f'{path}encoders.pkl')
    scaler = joblib.load(f'{path}scaler.pkl')
    return model, encoders, scaler


def is_model_trained(path='models/'):
    import os
    return (os.path.exists(f'{path}anomaly_model.pkl') and
            os.path.exists(f'{path}encoders.pkl') and
            os.path.exists(f'{path}scaler.pkl'))
