"""
Build a self-contained HTML demo file for AI05 SMS Spam Detection.
Embeds all charts, metrics, and a lightweight JS classifier.
"""
import json, base64
from pathlib import Path

OUT_DIR = Path("outputs/AI05")

# Load all data
with open(OUT_DIR / 'metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)
with open(OUT_DIR / 'charts.json', 'r', encoding='utf-8') as f:
    charts = json.load(f)
with open(OUT_DIR / 'browser_model.json', 'r', encoding='utf-8') as f:
    browser_model = json.load(f)

baseline = metrics['baseline']
improved = metrics['improved']
eda = metrics['eda_stats']
model_comp = metrics['model_comparison']
top_spam_words = metrics['top_spam_words']
top_ham_words = metrics['top_ham_words']

html = f'''<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🛡️ Lá Chắn AI – Phát hiện tin nhắn rác SMS</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --primary: #1a73e8;
  --danger: #d93025;
  --success: #0d904f;
  --bg: #f8f9fa;
  --card: #ffffff;
  --text: #202124;
  --text2: #5f6368;
  --shadow: 0 2px 8px rgba(0,0,0,0.08);
  --radius: 16px;
}}

body {{
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}

/* NAV */
.nav {{
  background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
  padding: 0;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}
.nav-inner {{
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  padding: 12px 24px;
  gap: 8px;
  overflow-x: auto;
}}
.nav-brand {{
  font-size: 20px;
  font-weight: 700;
  color: white;
  white-space: nowrap;
  margin-right: 16px;
}}
.nav a {{
  color: rgba(255,255,255,0.85);
  text-decoration: none;
  padding: 8px 16px;
  border-radius: 24px;
  font-size: 14px;
  font-weight: 500;
  white-space: nowrap;
  transition: all 0.2s;
}}
.nav a:hover, .nav a.active {{
  background: rgba(255,255,255,0.15);
  color: white;
}}

/* HERO */
.hero {{
  background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #01579b 100%);
  padding: 60px 24px 80px;
  text-align: center;
  color: white;
  position: relative;
  overflow: hidden;
}}
.hero::after {{
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  right: 0;
  height: 60px;
  background: var(--bg);
  border-radius: 50% 50% 0 0 / 100% 100% 0 0;
}}
.hero h1 {{
  font-size: 42px;
  font-weight: 800;
  margin-bottom: 12px;
  text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}
.hero p {{
  font-size: 18px;
  opacity: 0.9;
  max-width: 600px;
  margin: 0 auto;
}}
.hero-badges {{
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 24px;
  flex-wrap: wrap;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: 24px;
  font-size: 14px;
  font-weight: 600;
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
}}

/* CONTAINER */
.container {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px;
}}

/* SECTION */
.section {{
  padding: 60px 0 40px;
}}
.section-title {{
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.section-subtitle {{
  color: var(--text2);
  font-size: 16px;
  margin-bottom: 32px;
}}

/* CARDS */
.card {{
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 28px;
  margin-bottom: 24px;
  border: 1px solid #e8eaed;
  transition: box-shadow 0.2s;
}}
.card:hover {{
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}}

/* STAT GRID */
.stat-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}
.stat-card {{
  background: var(--card);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  box-shadow: var(--shadow);
  border: 1px solid #e8eaed;
}}
.stat-value {{
  font-size: 32px;
  font-weight: 800;
  margin: 4px 0;
}}
.stat-label {{
  font-size: 13px;
  color: var(--text2);
  font-weight: 500;
}}
.stat-card.ham .stat-value {{ color: var(--success); }}
.stat-card.spam .stat-value {{ color: var(--danger); }}
.stat-card.blue .stat-value {{ color: var(--primary); }}

/* CHART */
.chart-img {{
  width: 100%;
  border-radius: 12px;
  margin: 12px 0;
}}

/* TABLE */
.comparison-table {{
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 15px;
}}
.comparison-table th {{
  background: #f1f3f4;
  padding: 14px 16px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #dadce0;
}}
.comparison-table td {{
  padding: 12px 16px;
  border-bottom: 1px solid #e8eaed;
}}
.comparison-table tr:hover {{ background: #f8f9fa; }}
.improve {{ color: var(--success); font-weight: 700; }}
.worse {{ color: var(--danger); font-weight: 700; }}

/* CONFUSION MATRIX */
.cm-grid {{
  display: grid;
  grid-template-columns: auto 1fr 1fr;
  grid-template-rows: auto 1fr 1fr;
  gap: 2px;
  max-width: 300px;
  margin: 16px auto;
  font-size: 14px;
}}
.cm-header {{
  background: #e8eaed;
  padding: 10px;
  text-align: center;
  font-weight: 600;
  border-radius: 4px;
}}
.cm-cell {{
  padding: 16px;
  text-align: center;
  font-weight: 700;
  font-size: 20px;
  border-radius: 4px;
}}
.cm-tp {{ background: #c8e6c9; color: #1b5e20; }}
.cm-tn {{ background: #c8e6c9; color: #1b5e20; }}
.cm-fp {{ background: #ffcdd2; color: #b71c1c; }}
.cm-fn {{ background: #fff3e0; color: #e65100; }}

/* DEMO */
.demo-section {{
  background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
  border-radius: var(--radius);
  padding: 40px;
  margin: 32px 0;
}}
.demo-textarea {{
  width: 100%;
  min-height: 100px;
  padding: 16px;
  border: 2px solid #dadce0;
  border-radius: 12px;
  font-size: 16px;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.2s;
  background: white;
}}
.demo-textarea:focus {{
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(26,115,232,0.15);
}}
.demo-btn {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 14px 32px;
  border: none;
  border-radius: 24px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-top: 16px;
}}
.demo-btn-primary {{
  background: var(--primary);
  color: white;
}}
.demo-btn-primary:hover {{
  background: #1557b0;
  box-shadow: 0 4px 12px rgba(26,115,232,0.3);
  transform: translateY(-1px);
}}
.demo-btn-secondary {{
  background: white;
  color: var(--text);
  border: 2px solid #dadce0;
  margin-left: 8px;
}}
.demo-btn-secondary:hover {{
  background: #f1f3f4;
}}

/* RESULT */
.result-box {{
  margin-top: 24px;
  padding: 24px;
  border-radius: 12px;
  display: none;
  animation: slideUp 0.3s ease;
}}
@keyframes slideUp {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.result-ham {{
  background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
  border: 2px solid #4caf50;
}}
.result-spam {{
  background: linear-gradient(135deg, #ffebee, #ffcdd2);
  border: 2px solid #f44336;
}}
.result-label {{
  font-size: 28px;
  font-weight: 800;
  margin-bottom: 8px;
}}
.result-ham .result-label {{ color: #1b5e20; }}
.result-spam .result-label {{ color: #b71c1c; }}
.confidence-bar {{
  width: 100%;
  height: 12px;
  background: #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
  margin: 12px 0;
}}
.confidence-fill {{
  height: 100%;
  border-radius: 6px;
  transition: width 0.5s ease;
}}
.confidence-ham {{ background: linear-gradient(90deg, #4caf50, #2e7d32); }}
.confidence-spam {{ background: linear-gradient(90deg, #f44336, #c62828); }}

/* SAMPLE MESSAGES */
.sample-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
  margin-top: 16px;
}}
.sample-msg {{
  background: white;
  border-radius: 12px;
  padding: 14px;
  cursor: pointer;
  border: 2px solid #e8eaed;
  transition: all 0.2s;
  font-size: 14px;
}}
.sample-msg:hover {{
  border-color: var(--primary);
  box-shadow: 0 4px 12px rgba(26,115,232,0.15);
  transform: translateY(-2px);
}}
.sample-msg .tag {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 700;
  margin-bottom: 6px;
}}
.sample-msg .tag-spam {{ background: #ffcdd2; color: #b71c1c; }}
.sample-msg .tag-ham {{ background: #c8e6c9; color: #1b5e20; }}
.sample-msg .tag-tricky {{ background: #fff3e0; color: #e65100; }}

/* FEATURES TABLE */
.feat-table {{
  width: 100%;
  border-collapse: collapse;
}}
.feat-table th {{
  background: #f1f3f4;
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  font-size: 13px;
  border-bottom: 2px solid #dadce0;
}}
.feat-table td {{
  padding: 10px 14px;
  border-bottom: 1px solid #e8eaed;
  font-size: 14px;
}}
.feat-bar {{
  height: 20px;
  border-radius: 10px;
  display: inline-block;
  min-width: 4px;
}}

/* FOOTER */
.footer {{
  text-align: center;
  padding: 40px 24px;
  color: var(--text2);
  font-size: 14px;
  border-top: 1px solid #e8eaed;
  margin-top: 40px;
}}

/* GRID */
.grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}
@media (max-width: 768px) {{
  .grid-2 {{ grid-template-columns: 1fr; }}
  .hero h1 {{ font-size: 28px; }}
  .demo-section {{ padding: 20px; }}
  .nav-inner {{ padding: 8px 16px; }}
}}

/* TABS */
.tabs {{
  display: flex;
  gap: 4px;
  border-bottom: 2px solid #e8eaed;
  margin-bottom: 24px;
}}
.tab {{
  padding: 12px 24px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  color: var(--text2);
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  transition: all 0.2s;
}}
.tab:hover {{ color: var(--primary); }}
.tab.active {{
  color: var(--primary);
  border-bottom-color: var(--primary);
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* PILL */
.pill {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 600;
}}
.pill-green {{ background: #e8f5e9; color: #1b5e20; }}
.pill-red {{ background: #ffebee; color: #b71c1c; }}
.pill-blue {{ background: #e3f2fd; color: #0d47a1; }}
</style>
</head>
<body>

<!-- NAV -->
<nav class="nav">
  <div class="nav-inner">
    <span class="nav-brand">🛡️ AI05</span>
    <a href="#overview">Tổng quan</a>
    <a href="#eda">Phân tích</a>
    <a href="#baseline">Baseline</a>
    <a href="#improved">Cải tiến</a>
    <a href="#demo">Demo</a>
    <a href="#test">Kiểm thử</a>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <h1>🛡️ Lá Chắn AI</h1>
  <p>Hệ thống phát hiện và cảnh báo tin nhắn rác (SMS Spam) sử dụng Machine Learning</p>
  <div class="hero-badges">
    <span class="badge">📊 5,574 tin nhắn</span>
    <span class="badge">🎯 F1 Spam = {improved['spam_f1']:.1%}</span>
    <span class="badge">✅ FPR = {improved['ham_fpr']:.1%}</span>
    <span class="badge">🤖 {improved['model']}</span>
  </div>
</section>

<!-- OVERVIEW -->
<div class="container">
<section class="section" id="overview">
  <h2 class="section-title">📋 Tổng quan dự án</h2>
  <p class="section-subtitle">Mã đề AI05 – An toàn số · Smart Campus AI</p>

  <div class="card">
    <h3>🎯 Mục tiêu</h3>
    <p style="margin-top:8px">Xây dựng nguyên mẫu để người dùng kiểm tra một tin nhắn trước khi tương tác, phân biệt <strong>spam</strong> (tin rác) và <strong>ham</strong> (tin nhắn hợp lệ).</p>

    <h3 style="margin-top:20px">📂 Dữ liệu</h3>
    <p style="margin-top:8px"><strong>UCI SMS Spam Collection:</strong> 5,574 tin nhắn tiếng Anh, hai nhãn ham/spam.</p>

    <h3 style="margin-top:20px">📏 Chỉ số đánh giá</h3>
    <ul style="margin-top:8px; padding-left:24px">
      <li><strong>F1 lớp spam</strong> – Trung bình hài hòa giữa Precision và Recall của lớp spam</li>
      <li><strong>FPR (False Positive Rate)</strong> – Tỷ lệ tin ham bị phân nhầm thành spam</li>
    </ul>
  </div>
</section>

<!-- EDA -->
<section class="section" id="eda">
  <h2 class="section-title">📊 Phân tích dữ liệu (EDA)</h2>
  <p class="section-subtitle">Khám phá bộ dữ liệu trước khi xây dựng mô hình</p>

  <h3 style="margin-bottom: 16px;">1. Phân bố nhãn – Mất cân bằng dữ liệu</h3>
  <div class="stat-grid">
    <div class="stat-card ham">
      <div class="stat-label">Ham (Bình thường)</div>
      <div class="stat-value">{eda['ham']['count']:,}</div>
      <div class="stat-label">{eda['ham']['count']/(eda['ham']['count']+eda['spam']['count'])*100:.1f}%</div>
    </div>
    <div class="stat-card spam">
      <div class="stat-label">Spam (Rác)</div>
      <div class="stat-value">{eda['spam']['count']:,}</div>
      <div class="stat-label">{eda['spam']['count']/(eda['ham']['count']+eda['spam']['count'])*100:.1f}%</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-label">Tổng cộng</div>
      <div class="stat-value">{eda['ham']['count']+eda['spam']['count']:,}</div>
      <div class="stat-label">sau khi loại trùng</div>
    </div>
  </div>

  <div class="card">
    <p>⚠️ <strong>Dữ liệu mất cân bằng nghiêm trọng:</strong> Ham chiếm ~87.6%, Spam chỉ ~12.4%.
    Điều này có nghĩa nếu mô hình đoán tất cả là "ham" cũng đạt 87.6% accuracy – nhưng sẽ bỏ sót toàn bộ spam!</p>
    <img class="chart-img" src="data:image/png;base64,{charts['label_dist']}" alt="Label Distribution">
  </div>

  <h3 style="margin: 32px 0 16px;">2. Đặc trưng từ ngữ</h3>
  <div class="card">
    <h4>So sánh đặc trưng Ham vs Spam</h4>
    <table class="feat-table">
      <tr><th>Đặc trưng</th><th>Ham</th><th>Spam</th><th>Nhận xét</th></tr>
      <tr>
        <td>📏 Độ dài TB</td>
        <td>{eda['ham']['avg_length']:.0f} ký tự</td>
        <td>{eda['spam']['avg_length']:.0f} ký tự</td>
        <td><span class="pill pill-red">Spam dài gấp {eda['spam']['avg_length']/eda['ham']['avg_length']:.1f}x</span></td>
      </tr>
      <tr>
        <td>📝 Số từ TB</td>
        <td>{eda['ham']['avg_words']:.0f} từ</td>
        <td>{eda['spam']['avg_words']:.0f} từ</td>
        <td><span class="pill pill-red">Spam dài hơn</span></td>
      </tr>
      <tr>
        <td>🔠 % IN HOA</td>
        <td>{eda['ham']['avg_uppercase_ratio']*100:.1f}%</td>
        <td>{eda['spam']['avg_uppercase_ratio']*100:.1f}%</td>
        <td><span class="pill pill-red">Spam IN HOA nhiều hơn</span></td>
      </tr>
      <tr>
        <td>🔗 Có URL</td>
        <td>{eda['ham']['pct_has_url']:.1f}%</td>
        <td>{eda['spam']['pct_has_url']:.1f}%</td>
        <td><span class="pill pill-red">Spam có URL nhiều hơn rất nhiều</span></td>
      </tr>
      <tr>
        <td>📞 Có số điện thoại</td>
        <td>{eda['ham']['pct_has_phone']:.1f}%</td>
        <td>{eda['spam']['pct_has_phone']:.1f}%</td>
        <td><span class="pill pill-red">73.5% spam chứa SĐT!</span></td>
      </tr>
      <tr>
        <td>💰 Có ký hiệu tiền £/$</td>
        <td>{eda['ham']['pct_has_currency']:.1f}%</td>
        <td>{eda['spam']['pct_has_currency']:.1f}%</td>
        <td><span class="pill pill-red">1/3 spam nói về tiền</span></td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h4>Phân bố độ dài tin nhắn</h4>
    <img class="chart-img" src="data:image/png;base64,{charts['length_dist']}" alt="Message Length Distribution">
  </div>

  <div class="card">
    <h4>So sánh đặc trưng bằng biểu đồ</h4>
    <img class="chart-img" src="data:image/png;base64,{charts['feature_compare']}" alt="Feature Comparison">
  </div>

  <div class="card">
    <h4>Top từ xuất hiện nhiều nhất</h4>
    <img class="chart-img" src="data:image/png;base64,{charts['top_words']}" alt="Top Words">
    <div class="grid-2" style="margin-top:16px">
      <div>
        <h5>🔴 Top từ SPAM</h5>
        <p style="color:var(--text2); font-size:14px; margin-top:4px">
          {', '.join([f'<strong>{w}</strong> ({c})' for w,c in top_spam_words[:12]])}
        </p>
      </div>
      <div>
        <h5>🟢 Top từ HAM</h5>
        <p style="color:var(--text2); font-size:14px; margin-top:4px">
          {', '.join([f'<strong>{w}</strong> ({c})' for w,c in top_ham_words[:12]])}
        </p>
      </div>
    </div>
  </div>
</section>

<!-- BASELINE -->
<section class="section" id="baseline">
  <h2 class="section-title">🤖 Baseline Model</h2>
  <p class="section-subtitle">TF-IDF (unigram/bigram) + Multinomial Naive Bayes</p>

  <div class="stat-grid">
    <div class="stat-card blue">
      <div class="stat-label">Accuracy</div>
      <div class="stat-value">{baseline['accuracy']:.1%}</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-label">Spam F1</div>
      <div class="stat-value">{baseline['spam_f1']:.1%}</div>
    </div>
    <div class="stat-card ham">
      <div class="stat-label">Spam Precision</div>
      <div class="stat-value">{baseline['spam_precision']:.1%}</div>
    </div>
    <div class="stat-card spam">
      <div class="stat-label">Spam Recall</div>
      <div class="stat-value">{baseline['spam_recall']:.1%}</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h4>📊 Confusion Matrix</h4>
      <div class="cm-grid">
        <div></div>
        <div class="cm-header">Dự đoán<br>Ham</div>
        <div class="cm-header">Dự đoán<br>Spam</div>
        <div class="cm-header">Thực tế<br>Ham</div>
        <div class="cm-cell cm-tn">{baseline['confusion_matrix'][0][0]}</div>
        <div class="cm-cell cm-fp">{baseline['confusion_matrix'][0][1]}</div>
        <div class="cm-header">Thực tế<br>Spam</div>
        <div class="cm-cell cm-fn">{baseline['confusion_matrix'][1][0]}</div>
        <div class="cm-cell cm-tp">{baseline['confusion_matrix'][1][1]}</div>
      </div>
    </div>
    <div class="card">
      <h4>📝 Nhận xét Baseline</h4>
      <ul style="padding-left:20px; margin-top:8px; line-height:2">
        <li><strong>Precision = 100%:</strong> Không bao giờ nhầm ham thành spam</li>
        <li><strong>Recall chỉ {baseline['spam_recall']:.1%}:</strong> Bỏ sót {baseline['confusion_matrix'][1][0]}/{baseline['confusion_matrix'][1][0]+baseline['confusion_matrix'][1][1]} tin spam</li>
        <li>⚠️ <strong>Vấn đề:</strong> Mô hình quá "thận trọng", thà bỏ sót spam còn hơn nhầm ham</li>
        <li>🎯 <strong>Cần cải thiện:</strong> Tăng Recall mà giữ Precision cao</li>
      </ul>
    </div>
  </div>
</section>

<!-- IMPROVED -->
<section class="section" id="improved">
  <h2 class="section-title">🚀 Mô hình Cải tiến</h2>
  <p class="section-subtitle">3 hướng cải tiến: Tiền xử lý + Mô hình tốt hơn + Tối ưu ngưỡng</p>

  <div class="card">
    <h4>🔧 Các cải tiến đã thực hiện</h4>
    <div class="grid-2" style="margin-top:16px">
      <div>
        <h5>1. Tiền xử lý nâng cao</h5>
        <ul style="padding-left:20px; margin-top:4px; line-height:1.8; font-size:14px">
          <li>Thay URL → <code>URL</code></li>
          <li>Thay SĐT → <code>PHONENUMBER</code></li>
          <li>Thay tiền tệ → <code>MONEYAMT</code></li>
          <li>Thay email → <code>EMAILADDR</code></li>
          <li>Loại bỏ dấu câu, chuẩn hoá khoảng trắng</li>
        </ul>
      </div>
      <div>
        <h5>2. So sánh 3 mô hình</h5>
        <table class="feat-table" style="margin-top:4px">
          <tr><th>Mô hình</th><th>F1</th><th>FPR</th></tr>
          <tr><td>Naive Bayes</td><td>{model_comp['NaiveBayes']['f1']:.4f}</td><td>{model_comp['NaiveBayes']['fpr']:.4f}</td></tr>
          <tr><td>Logistic Regression</td><td>{model_comp['LogisticRegression']['f1']:.4f}</td><td>{model_comp['LogisticRegression']['fpr']:.4f}</td></tr>
          <tr style="background:#e3f2fd"><td><strong>LinearSVC ✅</strong></td><td><strong>{model_comp['LinearSVC']['f1']:.4f}</strong></td><td><strong>{model_comp['LinearSVC']['fpr']:.4f}</strong></td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <h4>3. Tối ưu ngưỡng quyết định</h4>
    <p style="margin-top:8px; color:var(--text2)">Thay vì ngưỡng mặc định 0.50, tìm ngưỡng tối ưu hoá F1-spam: <strong>{improved['threshold']:.2f}</strong></p>
    <img class="chart-img" src="data:image/png;base64,{charts['threshold_curve']}" alt="Threshold Optimization">
  </div>

  <h3 style="margin: 32px 0 16px;">📊 So sánh Baseline vs Cải tiến</h3>
  <div class="stat-grid">
    <div class="stat-card blue">
      <div class="stat-label">Accuracy</div>
      <div class="stat-value">{improved['accuracy']:.1%}</div>
      <div class="stat-label improve">↑ {(improved['accuracy']-baseline['accuracy'])*100:+.1f}%</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-label">Spam F1</div>
      <div class="stat-value">{improved['spam_f1']:.1%}</div>
      <div class="stat-label improve">↑ {(improved['spam_f1']-baseline['spam_f1'])*100:+.1f}%</div>
    </div>
    <div class="stat-card ham">
      <div class="stat-label">Spam Precision</div>
      <div class="stat-value">{improved['spam_precision']:.1%}</div>
    </div>
    <div class="stat-card ham">
      <div class="stat-label">Spam Recall</div>
      <div class="stat-value">{improved['spam_recall']:.1%}</div>
      <div class="stat-label improve">↑ {(improved['spam_recall']-baseline['spam_recall'])*100:+.1f}%</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h4>So sánh chi tiết</h4>
      <table class="comparison-table">
        <tr><th>Chỉ số</th><th>Baseline</th><th>Cải tiến</th><th>Thay đổi</th></tr>
        <tr>
          <td>Accuracy</td>
          <td>{baseline['accuracy']:.4f}</td>
          <td>{improved['accuracy']:.4f}</td>
          <td class="improve">↑ +{(improved['accuracy']-baseline['accuracy']):.4f}</td>
        </tr>
        <tr>
          <td><strong>Spam F1</strong></td>
          <td>{baseline['spam_f1']:.4f}</td>
          <td><strong>{improved['spam_f1']:.4f}</strong></td>
          <td class="improve"><strong>↑ +{(improved['spam_f1']-baseline['spam_f1']):.4f}</strong></td>
        </tr>
        <tr>
          <td>Spam Precision</td>
          <td>{baseline['spam_precision']:.4f}</td>
          <td>{improved['spam_precision']:.4f}</td>
          <td>{(improved['spam_precision']-baseline['spam_precision']):+.4f}</td>
        </tr>
        <tr>
          <td>Spam Recall</td>
          <td>{baseline['spam_recall']:.4f}</td>
          <td>{improved['spam_recall']:.4f}</td>
          <td class="improve">↑ +{(improved['spam_recall']-baseline['spam_recall']):.4f}</td>
        </tr>
        <tr>
          <td><strong>Ham FPR</strong></td>
          <td>{baseline['ham_fpr']:.4f}</td>
          <td><strong>{improved['ham_fpr']:.4f}</strong></td>
          <td class="improve">✅ Giữ nguyên</td>
        </tr>
      </table>
    </div>
    <div class="card">
      <img class="chart-img" src="data:image/png;base64,{charts['metrics_compare']}" alt="Metrics Comparison">
    </div>
  </div>

  <div class="card">
    <h4>Confusion Matrix – Baseline vs Cải tiến</h4>
    <img class="chart-img" src="data:image/png;base64,{charts['confusion_matrices']}" alt="Confusion Matrices Comparison">
    <p style="margin-top:12px; color:var(--text2);">
      ✅ Cải tiến bắt được thêm <strong>{improved['confusion_matrix'][1][1] - baseline['confusion_matrix'][1][1]}</strong> tin spam
      (từ {baseline['confusion_matrix'][1][1]} lên {improved['confusion_matrix'][1][1]}) trong khi
      vẫn giữ <strong>0 lỗi False Positive</strong> (không nhầm ham thành spam).
    </p>
  </div>
</section>

<!-- DEMO -->
<section class="section" id="demo">
  <h2 class="section-title">🖥️ Demo – Thử nghiệm trực tiếp</h2>
  <p class="section-subtitle">Nhập tin nhắn tiếng Anh để kiểm tra spam/ham ngay trên trình duyệt</p>

  <div class="demo-section">
    <h3 style="margin-bottom:12px">✍️ Nhập tin nhắn cần kiểm tra</h3>
    <textarea class="demo-textarea" id="msgInput" placeholder="Ví dụ: Congratulations! You have been selected to win a FREE prize worth £1000! Call 09061234567 NOW"></textarea>
    <div>
      <button class="demo-btn demo-btn-primary" onclick="classifyMessage()">🔍 Kiểm tra ngay</button>
      <button class="demo-btn demo-btn-secondary" onclick="clearResult()">🗑️ Xóa</button>
    </div>

    <div class="result-box" id="resultBox">
      <div class="result-label" id="resultLabel"></div>
      <div id="resultDesc" style="margin-bottom:12px; color:var(--text2)"></div>
      <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:600">
        <span>Ham (Bình thường)</span>
        <span>Spam (Rác)</span>
      </div>
      <div class="confidence-bar">
        <div class="confidence-fill" id="confidenceFill"></div>
      </div>
      <div id="confidenceText" style="text-align:center; font-weight:600; font-size:14px"></div>
      <div id="topFeatures" style="margin-top:16px; font-size:13px; color:var(--text2)"></div>
    </div>
  </div>

  <h3 style="margin: 32px 0 12px">📝 Tin nhắn mẫu – Click để thử</h3>
  <div class="sample-grid">
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-spam">SPAM</span>
      <p>WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-spam">SPAM</span>
      <p>Free entry in 2 a weekly comp to win FA Cup final tkts. Text FA to 87121 to receive entry.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-spam">SPAM</span>
      <p>URGENT! Your mobile number has been awarded a £2000 bonus prize. Call 09066362206 NOW!</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-ham">HAM</span>
      <p>Hey, what time are we meeting for lunch today?</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-ham">HAM</span>
      <p>Can you pick up some milk on your way home please?</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-ham">HAM</span>
      <p>The meeting has been rescheduled to 3pm tomorrow. See you there!</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>Your free trial expires tomorrow. Reply STOP to cancel subscription.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>Congrats on your promotion! Let's celebrate tonight at the restaurant.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>I won the game last night, it was amazing! You should have seen it.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>Call me when you're free, I need to talk urgently about the project.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>You have won best employee of the month! Come collect your certificate from HR.</p>
    </div>
    <div class="sample-msg" onclick="tryMessage(this)">
      <span class="tag tag-tricky">DỄ NHẦM</span>
      <p>REMINDER: Your dental appointment is confirmed for tomorrow at 10am. Reply YES to confirm.</p>
    </div>
  </div>
</section>

<!-- TEST RESULTS -->
<section class="section" id="test">
  <h2 class="section-title">🧪 Kiểm thử với tin nhắn mới</h2>
  <p class="section-subtitle">Kết quả kiểm thử mô hình với các tin nhắn chưa từng thấy trong dữ liệu huấn luyện</p>

  <div class="card">
    <p style="color:var(--text2); margin-bottom:16px">Bảng dưới đây sẽ tự động điền khi bạn click các tin nhắn mẫu ở phần Demo hoặc nhập tin nhắn mới.</p>
    <table class="comparison-table" id="testTable">
      <thead>
        <tr>
          <th style="width:50%">Tin nhắn</th>
          <th>Nhãn thật</th>
          <th>Dự đoán</th>
          <th>P(spam)</th>
          <th>Kết quả</th>
        </tr>
      </thead>
      <tbody id="testTableBody">
      </tbody>
    </table>
    <button class="demo-btn demo-btn-secondary" onclick="clearTestTable()" style="margin-top:12px">🗑️ Xóa bảng</button>
    <button class="demo-btn demo-btn-primary" onclick="runAllSamples()" style="margin-top:12px">▶️ Chạy tất cả mẫu</button>
  </div>
</section>

</div>

<!-- FOOTER -->
<div class="footer">
  <p><strong>🛡️ AI05 – Lá Chắn AI</strong> · Smart Campus AI</p>
  <p>Dữ liệu: UCI SMS Spam Collection (CC BY 4.0) · Mô hình: {improved['model']} + TF-IDF</p>
  <p style="margin-top:8px; font-size:12px; color:#999">Cảnh báo tham khảo, không chắc chắn là lừa đảo. Chỉ dùng cho mục đích học thuật.</p>
</div>

<script>
// ============================================================
// BROWSER-BASED SPAM CLASSIFIER
// ============================================================
const MODEL_DATA = BROWSER_MODEL_PLACEHOLDER;

function preprocessText(text) {{
  let t = text.toLowerCase();
  t = t.replace(/http[s]?:\\/\\/\\S+|www\\.\\S+/g, ' URL ');
  t = t.replace(/\\b\\d{{5,}}\\b/g, ' PHONENUMBER ');
  t = t.replace(/[£$€]\\s*\\d+[\\d,.]*/g, ' MONEYAMT ');
  t = t.replace(/\\S+@\\S+/g, ' EMAILADDR ');
  t = t.replace(/[^\\w\\s]/g, ' ');
  t = t.replace(/\\s+/g, ' ').trim();
  return t;
}}

function getTokens(text) {{
  const words = text.split(' ').filter(w => w.length > 0);
  const tokens = [...words];
  for (let i = 0; i < words.length - 1; i++) {{
    tokens.push(words[i] + ' ' + words[i+1]);
  }}
  return tokens;
}}

function classify(text) {{
  const processed = preprocessText(text);
  const tokens = getTokens(processed);
  const vocab = MODEL_DATA.vocab;

  let score = 0;
  let totalWeight = 0;
  const matchedFeatures = [];

  const tokenCounts = {{}};
  tokens.forEach(t => {{ tokenCounts[t] = (tokenCounts[t] || 0) + 1; }});

  for (const [token, count] of Object.entries(tokenCounts)) {{
    if (vocab[token]) {{
      const contrib = vocab[token].score * vocab[token].idf * Math.log(1 + count);
      score += contrib;
      totalWeight += Math.abs(contrib);
      matchedFeatures.push({{ token, score: contrib, idf: vocab[token].idf }});
    }}
  }}

  // Normalize to probability-like score using sigmoid
  const sigmoid = 1 / (1 + Math.exp(-score * 0.5));
  const isSpam = sigmoid >= MODEL_DATA.threshold;

  // Sort features by absolute contribution
  matchedFeatures.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

  return {{
    label: isSpam ? 'spam' : 'ham',
    probability: sigmoid,
    topFeatures: matchedFeatures.slice(0, 8),
    rawScore: score
  }};
}}

function classifyMessage() {{
  const text = document.getElementById('msgInput').value.trim();
  if (!text) {{ alert('Vui lòng nhập tin nhắn!'); return; }}

  const result = classify(text);
  showResult(result);
}}

function showResult(result) {{
  const box = document.getElementById('resultBox');
  const label = document.getElementById('resultLabel');
  const desc = document.getElementById('resultDesc');
  const fill = document.getElementById('confidenceFill');
  const confText = document.getElementById('confidenceText');
  const features = document.getElementById('topFeatures');

  box.style.display = 'block';
  box.className = 'result-box ' + (result.label === 'spam' ? 'result-spam' : 'result-ham');

  if (result.label === 'spam') {{
    label.textContent = '⚠️ SPAM – Tin nhắn rác!';
    desc.textContent = 'Tin nhắn này có dấu hiệu là spam/quảng cáo. Hãy cẩn thận trước khi tương tác.';
    fill.className = 'confidence-fill confidence-spam';
    fill.style.width = (result.probability * 100) + '%';
  }} else {{
    label.textContent = '✅ HAM – Tin nhắn bình thường';
    desc.textContent = 'Tin nhắn này có vẻ hợp lệ và an toàn.';
    fill.className = 'confidence-fill confidence-ham';
    fill.style.width = ((1 - result.probability) * 100) + '%';
  }}

  confText.textContent = `P(spam) = ${{(result.probability * 100).toFixed(1)}}% | Ngưỡng: ${{(MODEL_DATA.threshold * 100).toFixed(0)}}%`;

  if (result.topFeatures.length > 0) {{
    const spamWords = result.topFeatures.filter(f => f.score > 0).map(f => f.token);
    const hamWords = result.topFeatures.filter(f => f.score < 0).map(f => f.token);
    let html = '<strong>Từ khóa ảnh hưởng:</strong> ';
    if (spamWords.length) html += '🔴 Spam: ' + spamWords.map(w => `<code>${{w}}</code>`).join(', ') + ' ';
    if (hamWords.length) html += '🟢 Ham: ' + hamWords.map(w => `<code>${{w}}</code>`).join(', ');
    features.innerHTML = html;
  }} else {{
    features.innerHTML = '<em>Không tìm thấy từ khóa đặc trưng trong từ điển mô hình.</em>';
  }}
}}

function clearResult() {{
  document.getElementById('msgInput').value = '';
  document.getElementById('resultBox').style.display = 'none';
}}

function tryMessage(el) {{
  const text = el.querySelector('p').textContent;
  document.getElementById('msgInput').value = text;
  classifyMessage();

  // Determine expected label
  const tagEl = el.querySelector('.tag');
  let expected = '?';
  if (tagEl.classList.contains('tag-spam')) expected = 'spam';
  else if (tagEl.classList.contains('tag-ham')) expected = 'ham';
  else expected = tagEl.textContent.trim();

  const result = classify(text);
  addTestRow(text, expected, result);

  // Scroll to result
  document.getElementById('resultBox').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

function addTestRow(text, expected, result) {{
  const tbody = document.getElementById('testTableBody');
  const tr = document.createElement('tr');
  const shortText = text.length > 60 ? text.substring(0, 57) + '...' : text;
  const isCorrect = expected === '?' || expected.toLowerCase().includes(result.label);
  tr.innerHTML = `
    <td title="${{text.replace(/"/g, '&quot;')}}" style="font-size:13px">${{shortText}}</td>
    <td>${{expected}}</td>
    <td><span class="pill ${{result.label==='spam'?'pill-red':'pill-green'}}">${{result.label}}</span></td>
    <td>${{(result.probability * 100).toFixed(1)}}%</td>
    <td>${{isCorrect ? '✅' : '❌'}}</td>
  `;
  tbody.appendChild(tr);
}}

function clearTestTable() {{
  document.getElementById('testTableBody').innerHTML = '';
}}

function runAllSamples() {{
  clearTestTable();
  document.querySelectorAll('.sample-msg').forEach(el => {{
    const text = el.querySelector('p').textContent;
    const tagEl = el.querySelector('.tag');
    let expected = '?';
    if (tagEl.classList.contains('tag-spam')) expected = 'spam';
    else if (tagEl.classList.contains('tag-ham')) expected = 'ham';
    else expected = tagEl.textContent.trim();
    const result = classify(text);
    addTestRow(text, expected, result);
  }});
  document.getElementById('test').scrollIntoView({{ behavior: 'smooth' }});
}}

// Smooth scroll for nav links
document.querySelectorAll('.nav a').forEach(a => {{
  a.addEventListener('click', e => {{
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }});
}});

// Active nav on scroll
window.addEventListener('scroll', () => {{
  const sections = document.querySelectorAll('.section');
  let current = '';
  sections.forEach(s => {{
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  }});
  document.querySelectorAll('.nav a').forEach(a => {{
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  }});
}});
</script>
</body>
</html>'''

# Inject the browser model data
browser_model_json = json.dumps(browser_model, ensure_ascii=False)
html = html.replace('BROWSER_MODEL_PLACEHOLDER', browser_model_json)

# Save
out_path = Path("AI05_Demo.html")
out_path.write_text(html, encoding='utf-8')
print(f"HTML demo saved: {out_path.resolve()} ({out_path.stat().st_size/1024:.0f} KB)")
print("Open this file in your browser!")
