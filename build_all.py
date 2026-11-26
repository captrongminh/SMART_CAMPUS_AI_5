"""
AI05 - Lá Chắn AI: SMS Spam Detection
Complete pipeline: EDA + Baseline + Improved Model + Export for Browser
"""
import json, re, string, os, base64, io
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/AI05/SMSSpamCollection"
OUT_DIR = Path("outputs/AI05")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading and cleaning data...")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'text'],
                 encoding='latin-1', engine='python', quoting=3)  # QUOTE_NONE to handle quotes in SMS text
df = df[df.label.isin(['ham', 'spam'])].dropna().copy()

print(f"Total messages: {len(df)}")
print(f"Ham: {(df.label=='ham').sum()}, Spam: {(df.label=='spam').sum()}")
print(f"Spam ratio: {(df.label=='spam').mean()*100:.1f}%")

# Remove duplicates and ambiguous
df['normalized'] = df.text.astype(str).str.strip().str.lower()
bad = df.groupby('normalized').label.nunique()
ambiguous = set(bad[bad > 1].index)
df = df[~df.normalized.isin(ambiguous)].drop_duplicates('normalized')
print(f"After dedup: {len(df)} messages")

# ============================================================
# 2. EDA - Exploratory Data Analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis...")
print("=" * 60)

# Label distribution
label_counts = df.label.value_counts()
print(f"\nLabel distribution:")
print(f"  Ham:  {label_counts.get('ham', 0)} ({label_counts.get('ham', 0)/len(df)*100:.1f}%)")
print(f"  Spam: {label_counts.get('spam', 0)} ({label_counts.get('spam', 0)/len(df)*100:.1f}%)")

# Message length analysis
df['msg_length'] = df.text.str.len()
df['word_count'] = df.text.str.split().str.len()
df['has_url'] = df.text.str.contains(r'http[s]?://|www\.', case=False, regex=True).astype(int)
df['has_phone'] = df.text.str.contains(r'\b\d{5,}\b', regex=True).astype(int)
df['has_currency'] = df.text.str.contains(r'[£$€]', regex=True).astype(int)
df['uppercase_ratio'] = df.text.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
df['exclamation_count'] = df.text.str.count('!')

eda_stats = {}
for label in ['ham', 'spam']:
    subset = df[df.label == label]
    eda_stats[label] = {
        'count': int(len(subset)),
        'avg_length': float(subset.msg_length.mean()),
        'avg_words': float(subset.word_count.mean()),
        'avg_uppercase_ratio': float(subset.uppercase_ratio.mean()),
        'avg_exclamation': float(subset.exclamation_count.mean()),
        'pct_has_url': float(subset.has_url.mean() * 100),
        'pct_has_phone': float(subset.has_phone.mean() * 100),
        'pct_has_currency': float(subset.has_currency.mean() * 100),
    }
    print(f"\n{label.upper()}:")
    for k, v in eda_stats[label].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# Top words per class
def get_top_words(texts, n=30):
    words = ' '.join(texts).lower().split()
    stop = {'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
            'he','him','his','she','her','hers','it','its','they','them','their',
            'what','which','who','whom','this','that','am','is','are','was','were',
            'be','been','being','have','has','had','having','do','does','did','doing',
            'a','an','the','and','but','if','or','because','as','until','while',
            'of','at','by','for','with','about','against','between','through','during',
            'before','after','above','below','to','from','up','down','in','out','on',
            'off','over','under','again','further','then','once','here','there','when',
            'where','why','how','all','both','each','few','more','most','other','some',
            'such','no','nor','not','only','own','same','so','than','too','very',
            's','t','can','will','just','don','should','now','d','ll','m','o','re',
            've','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
            'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
            'won','wouldn'}
    words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation) and w.strip(string.punctuation) not in stop and len(w.strip(string.punctuation)) > 1]
    return Counter(words).most_common(n)

ham_words = get_top_words(df[df.label == 'ham'].text)
spam_words = get_top_words(df[df.label == 'spam'].text)

print(f"\nTop 15 SPAM words: {[w for w,c in spam_words[:15]]}")
print(f"Top 15 HAM words:  {[w for w,c in ham_words[:15]]}")

# ============================================================
# 3. GENERATE CHARTS (save as base64 for HTML embedding)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Generating charts...")
print("=" * 60)

charts = {}

# Chart 1: Label distribution
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#4CAF50', '#F44336']
bars = ax.bar(['Ham (Bình thường)', 'Spam (Rác)'],
              [label_counts.get('ham', 0), label_counts.get('spam', 0)],
              color=colors, edgecolor='white', linewidth=2)
for bar, count in zip(bars, [label_counts.get('ham', 0), label_counts.get('spam', 0)]):
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Phân bố nhãn Ham vs Spam', fontsize=14, fontweight='bold')
ax.set_ylabel('Số lượng tin nhắn')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['label_dist'] = base64.b64encode(buf.getvalue()).decode()

# Chart 2: Message length distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, label in enumerate(['ham', 'spam']):
    subset = df[df.label == label]
    color = '#4CAF50' if label == 'ham' else '#F44336'
    axes[i].hist(subset.msg_length, bins=50, color=color, alpha=0.7, edgecolor='white')
    axes[i].axvline(subset.msg_length.mean(), color='black', linestyle='--', label=f'TB={subset.msg_length.mean():.0f}')
    axes[i].set_title(f'Độ dài tin {label.upper()}', fontweight='bold')
    axes[i].set_xlabel('Số ký tự')
    axes[i].set_ylabel('Số lượng')
    axes[i].legend()
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['length_dist'] = base64.b64encode(buf.getvalue()).decode()

# Chart 3: Top words comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_n = 15
# Spam words
sw = spam_words[:top_n][::-1]
axes[0].barh([w for w, c in sw], [c for w, c in sw], color='#F44336', alpha=0.8)
axes[0].set_title(f'Top {top_n} từ trong SPAM', fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
# Ham words
hw = ham_words[:top_n][::-1]
axes[1].barh([w for w, c in hw], [c for w, c in hw], color='#4CAF50', alpha=0.8)
axes[1].set_title(f'Top {top_n} từ trong HAM', fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['top_words'] = base64.b64encode(buf.getvalue()).decode()

# Chart 4: Feature comparison (spider/grouped bar)
fig, ax = plt.subplots(figsize=(10, 5))
features = ['Độ dài TB', 'Số từ TB', '% IN HOA', '% có URL', '% có SĐT', '% có £/$']
ham_vals = [eda_stats['ham']['avg_length']/10, eda_stats['ham']['avg_words'],
            eda_stats['ham']['avg_uppercase_ratio']*100,
            eda_stats['ham']['pct_has_url'], eda_stats['ham']['pct_has_phone'],
            eda_stats['ham']['pct_has_currency']]
spam_vals = [eda_stats['spam']['avg_length']/10, eda_stats['spam']['avg_words'],
             eda_stats['spam']['avg_uppercase_ratio']*100,
             eda_stats['spam']['pct_has_url'], eda_stats['spam']['pct_has_phone'],
             eda_stats['spam']['pct_has_currency']]
x = np.arange(len(features))
w = 0.35
ax.bar(x - w/2, ham_vals, w, label='Ham', color='#4CAF50', alpha=0.8)
ax.bar(x + w/2, spam_vals, w, label='Spam', color='#F44336', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=15, ha='right')
ax.set_title('So sánh đặc trưng Ham vs Spam', fontsize=14, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['feature_compare'] = base64.b64encode(buf.getvalue()).decode()

print("Charts generated: ", list(charts.keys()))

# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Splitting data...")
print("=" * 60)

tr, va = train_test_split(df, test_size=0.2, stratify=df.label, random_state=2026)
print(f"Train: {len(tr)} | Validation: {len(va)}")

# ============================================================
# 5. BASELINE MODEL: TF-IDF + Multinomial Naive Bayes
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training BASELINE model (TF-IDF + Naive Bayes)...")
print("=" * 60)

baseline_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    MultinomialNB(alpha=0.5)
)
baseline_model.fit(tr.text, tr.label)
baseline_pred = baseline_model.predict(va.text)

ham_mask = va.label.to_numpy() == 'ham'
baseline_fp = ((baseline_pred == 'spam') & ham_mask).sum()

baseline_metrics = {
    'accuracy': float(accuracy_score(va.label, baseline_pred)),
    'spam_f1': float(f1_score(va.label, baseline_pred, pos_label='spam')),
    'spam_precision': float(precision_score(va.label, baseline_pred, pos_label='spam')),
    'spam_recall': float(recall_score(va.label, baseline_pred, pos_label='spam')),
    'ham_fpr': float(baseline_fp / max(1, ham_mask.sum())),
    'confusion_matrix': confusion_matrix(va.label, baseline_pred, labels=['ham', 'spam']).tolist(),
}

print(f"  Accuracy:        {baseline_metrics['accuracy']:.4f}")
print(f"  Spam F1:         {baseline_metrics['spam_f1']:.4f}")
print(f"  Spam Precision:  {baseline_metrics['spam_precision']:.4f}")
print(f"  Spam Recall:     {baseline_metrics['spam_recall']:.4f}")
print(f"  Ham FPR:         {baseline_metrics['ham_fpr']:.4f}")
print(f"  Confusion Matrix: {baseline_metrics['confusion_matrix']}")

# ============================================================
# 6. IMPROVED MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training IMPROVED model...")
print("=" * 60)

# --- Preprocessing improvements ---
def preprocess_text(text):
    """Enhanced text preprocessing"""
    text = str(text)
    # Lowercase
    text_lower = text.lower()
    # Feature extraction before lowering
    has_url = 1 if re.search(r'http[s]?://|www\.', text_lower) else 0
    has_phone = 1 if re.search(r'\b\d{5,}\b', text) else 0
    has_currency = 1 if re.search(r'[£$€]', text) else 0
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count('!')

    # Replace URLs
    text_lower = re.sub(r'http[s]?://\S+|www\.\S+', ' URL ', text_lower)
    # Replace phone numbers
    text_lower = re.sub(r'\b\d{5,}\b', ' PHONENUMBER ', text_lower)
    # Replace currency amounts
    text_lower = re.sub(r'[£$€]\s*\d+[\d,.]*', ' MONEYAMT ', text_lower)
    # Replace email
    text_lower = re.sub(r'\S+@\S+', ' EMAILADDR ', text_lower)
    # Remove punctuation but keep spaces
    text_lower = re.sub(r'[^\w\s]', ' ', text_lower)
    # Collapse whitespace
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()

    return text_lower

tr_processed = tr.text.apply(preprocess_text)
va_processed = va.text.apply(preprocess_text)

# --- Better TF-IDF ---
tfidf_improved = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=8000,
    strip_accents='unicode',
)

X_tr = tfidf_improved.fit_transform(tr_processed)
X_va = tfidf_improved.transform(va_processed)

# --- Try multiple models ---
models_to_try = {
    'NaiveBayes': MultinomialNB(alpha=0.3),
    'LogisticRegression': LogisticRegression(C=5, max_iter=1000, class_weight='balanced', solver='liblinear'),
    'LinearSVC': CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000), cv=3),
}

best_model_name = None
best_f1 = 0
best_model = None
model_results = {}

for name, model in models_to_try.items():
    model.fit(X_tr, tr.label)
    pred = model.predict(X_va)
    f1 = f1_score(va.label, pred, pos_label='spam')
    prec = precision_score(va.label, pred, pos_label='spam')
    rec = recall_score(va.label, pred, pos_label='spam')
    fp = ((pred == 'spam') & ham_mask).sum()
    fpr = fp / max(1, ham_mask.sum())

    model_results[name] = {
        'f1': float(f1), 'precision': float(prec),
        'recall': float(rec), 'fpr': float(fpr),
        'accuracy': float(accuracy_score(va.label, pred)),
        'confusion_matrix': confusion_matrix(va.label, pred, labels=['ham', 'spam']).tolist(),
    }
    print(f"  {name}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, FPR={fpr:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"\n  Best model: {best_model_name} (F1={best_f1:.4f})")

# --- Threshold optimization ---
print("\n  Optimizing threshold...")
proba = best_model.predict_proba(X_va)
spam_idx = list(best_model.classes_).index('spam')
spam_proba = proba[:, spam_idx]

best_threshold = 0.5
best_threshold_f1 = 0
threshold_results = []

for t in np.arange(0.3, 0.8, 0.02):
    t_pred = np.where(spam_proba >= t, 'spam', 'ham')
    t_f1 = f1_score(va.label, t_pred, pos_label='spam')
    t_prec = precision_score(va.label, t_pred, pos_label='spam', zero_division=0)
    t_fp = ((t_pred == 'spam') & ham_mask).sum()
    t_fpr = t_fp / max(1, ham_mask.sum())
    threshold_results.append({'threshold': float(t), 'f1': float(t_f1), 'precision': float(t_prec), 'fpr': float(t_fpr)})
    if t_f1 > best_threshold_f1:
        best_threshold_f1 = t_f1
        best_threshold = t

print(f"  Best threshold: {best_threshold:.2f} (F1={best_threshold_f1:.4f})")

# Final improved predictions
final_pred = np.where(spam_proba >= best_threshold, 'spam', 'ham')
final_fp = ((final_pred == 'spam') & ham_mask).sum()

improved_metrics = {
    'model': best_model_name,
    'threshold': float(best_threshold),
    'accuracy': float(accuracy_score(va.label, final_pred)),
    'spam_f1': float(f1_score(va.label, final_pred, pos_label='spam')),
    'spam_precision': float(precision_score(va.label, final_pred, pos_label='spam')),
    'spam_recall': float(recall_score(va.label, final_pred, pos_label='spam')),
    'ham_fpr': float(final_fp / max(1, ham_mask.sum())),
    'confusion_matrix': confusion_matrix(va.label, final_pred, labels=['ham', 'spam']).tolist(),
}

print(f"\n  IMPROVED RESULTS:")
print(f"  Accuracy:        {improved_metrics['accuracy']:.4f}")
print(f"  Spam F1:         {improved_metrics['spam_f1']:.4f}")
print(f"  Spam Precision:  {improved_metrics['spam_precision']:.4f}")
print(f"  Spam Recall:     {improved_metrics['spam_recall']:.4f}")
print(f"  Ham FPR:         {improved_metrics['ham_fpr']:.4f}")

# ============================================================
# 7. GENERATE COMPARISON CHART
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Generating comparison charts...")
print("=" * 60)

# Confusion matrix charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (title, cm) in enumerate([('Baseline (NB)', baseline_metrics['confusion_matrix']),
                                   ('Cải tiến (' + best_model_name + ')', improved_metrics['confusion_matrix'])]):
    cm_arr = np.array(cm)
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                annot_kws={'size': 16})
    axes[i].set_title(title, fontsize=13, fontweight='bold')
    axes[i].set_ylabel('Thực tế')
    axes[i].set_xlabel('Dự đoán')
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['confusion_matrices'] = base64.b64encode(buf.getvalue()).decode()

# Metric comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics_names = ['Accuracy', 'Spam F1', 'Spam Precision', 'Spam Recall']
baseline_vals = [baseline_metrics['accuracy'], baseline_metrics['spam_f1'],
                 baseline_metrics['spam_precision'], baseline_metrics['spam_recall']]
improved_vals = [improved_metrics['accuracy'], improved_metrics['spam_f1'],
                 improved_metrics['spam_precision'], improved_metrics['spam_recall']]
x = np.arange(len(metrics_names))
w = 0.35
b1 = ax.bar(x - w/2, baseline_vals, w, label='Baseline', color='#90CAF9', edgecolor='white')
b2 = ax.bar(x + w/2, improved_vals, w, label='Cải tiến', color='#2196F3', edgecolor='white')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0.8, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_title('So sánh Baseline vs Cải tiến', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['metrics_compare'] = base64.b64encode(buf.getvalue()).decode()

# Threshold curve
fig, ax = plt.subplots(figsize=(8, 4))
thresholds = [r['threshold'] for r in threshold_results]
f1s = [r['f1'] for r in threshold_results]
fprs = [r['fpr'] for r in threshold_results]
ax.plot(thresholds, f1s, 'b-o', markersize=3, label='F1 Spam')
ax.plot(thresholds, fprs, 'r-s', markersize=3, label='FPR (Ham bị nhầm)')
ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, label=f'Ngưỡng tối ưu = {best_threshold:.2f}')
ax.set_xlabel('Ngưỡng quyết định (Threshold)')
ax.set_ylabel('Giá trị')
ax.set_title('Ảnh hưởng của Ngưỡng đến F1 và FPR', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close()
charts['threshold_curve'] = base64.b64encode(buf.getvalue()).decode()

# ============================================================
# 8. EXPORT MODEL FOR BROWSER (JavaScript)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Exporting model for browser...")
print("=" * 60)

# Get vocabulary and IDF weights
vocab = tfidf_improved.vocabulary_
idf = tfidf_improved.idf_.tolist()
feature_names = tfidf_improved.get_feature_names_out().tolist()

# Get model parameters
if best_model_name == 'LogisticRegression':
    model_params = {
        'type': 'logistic_regression',
        'coef': best_model.coef_.tolist(),
        'intercept': best_model.intercept_.tolist(),
        'classes': best_model.classes_.tolist(),
    }
elif best_model_name == 'LinearSVC':
    # CalibratedClassifierCV wraps the SVC
    # We'll export as logistic since calibrated
    # For browser, we need to use the calibrated predict_proba approach
    # Simpler: just export the vocabulary + use a simple scoring approach
    base_svc = best_model.calibrated_classifiers_[0].estimator
    model_params = {
        'type': 'linear_svc_calibrated',
        'coef': base_svc.coef_.tolist(),
        'intercept': base_svc.intercept_.tolist(),
        'classes': best_model.classes_.tolist(),
    }
elif best_model_name == 'NaiveBayes':
    model_params = {
        'type': 'naive_bayes',
        'feature_log_prob': best_model.feature_log_prob_.tolist(),
        'class_log_prior': best_model.class_log_prior_.tolist(),
        'classes': best_model.classes_.tolist(),
    }

# Build a simpler export: top N most discriminative features with scores
# This is more practical for browser-based classification
print("  Building browser-compatible model...")

# Calculate feature importance (works for all linear models)
if hasattr(best_model, 'coef_'):
    feature_scores = best_model.coef_[0] if best_model.coef_.shape[0] == 1 else best_model.coef_[list(best_model.classes_).index('spam')]
elif best_model_name == 'LinearSVC':
    base = best_model.calibrated_classifiers_[0].estimator
    feature_scores = base.coef_[0]
else:
    feature_scores = best_model.feature_log_prob_[list(best_model.classes_).index('spam')] - best_model.feature_log_prob_[list(best_model.classes_).index('ham')]

# Export top features for lightweight browser model
top_k = min(3000, len(feature_names))
top_indices = np.argsort(np.abs(feature_scores))[-top_k:]

browser_vocab = {}
for idx in top_indices:
    browser_vocab[feature_names[idx]] = {
        'idf': float(idf[idx]),
        'score': float(feature_scores[idx]),
    }

browser_model = {
    'vocab': browser_vocab,
    'threshold': float(best_threshold),
    'model_type': best_model_name,
    'preprocessing': 'lowercase, url/phone/money/email replacement, punctuation removal',
}

# Export for browser
browser_model_path = OUT_DIR / 'browser_model.json'
with open(browser_model_path, 'w', encoding='utf-8') as f:
    json.dump(browser_model, f, ensure_ascii=False)
print(f"  Browser model saved: {browser_model_path} ({browser_model_path.stat().st_size/1024:.1f} KB)")

# ============================================================
# 9. SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Saving all results...")
print("=" * 60)

# Save metrics
all_results = {
    'baseline': baseline_metrics,
    'improved': improved_metrics,
    'model_comparison': model_results,
    'threshold_analysis': threshold_results,
    'eda_stats': eda_stats,
    'top_spam_words': spam_words[:20],
    'top_ham_words': ham_words[:20],
}

with open(OUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Save charts
with open(OUT_DIR / 'charts.json', 'w', encoding='utf-8') as f:
    json.dump(charts, f)

# Save models
joblib.dump(baseline_model, OUT_DIR / 'baseline_model.joblib')
joblib.dump({'tfidf': tfidf_improved, 'model': best_model, 'threshold': best_threshold, 'preprocess': preprocess_text},
            OUT_DIR / 'improved_model.joblib')

# Save some demo messages
demo_msgs = va[['text', 'label']].sample(10, random_state=42)
demo_msgs.to_csv(OUT_DIR / 'demo_messages.csv', index=False)

print("  All files saved to:", OUT_DIR)
print("\n" + "=" * 60)
print("ALL DONE! Now run: python build_html.py")
print("=" * 60)
