# 🛡️ AI05 – Lá Chắn AI: Phát hiện và cảnh báo tin nhắn rác (SMS Spam)

Nguyên mẫu phân loại tin nhắn tiếng Anh thành **Ham** (hợp lệ) và **Spam** (tin nhắn rác), kèm mức độ tin cậy và giao diện web chạy trực tiếp trên trình duyệt.

## 📊 Kết quả chính (trên tập Validation)

| Mô hình | Accuracy | Spam F1 | Spam Precision | Spam Recall | Ham FPR |
|---|---:|---:|---:|---:|---:|
| Baseline (TF-IDF + MultinomialNB) | 97.00% | 86.22% | 100.00% | 75.78% | 0.00% |
| **Cải tiến (Tiền xử lý + LinearSVC + Ngưỡng 0.54)** | **99.13%** | **96.36%** | **100.00%** | **92.97%** | **0.00%** |

## 📁 Cấu trúc thư mục

- `index.html` / `AI05_Demo.html`: Giao diện web tương tác (chạy trực tiếp trên trình duyệt / GitHub Pages, không cần backend).
- `train.py`: Mã baseline ban đầu (TF-IDF + Multinomial Naive Bayes).
- `demo.py`: Demo dòng lệnh (CLI) của baseline.
- `build_all.py`: Script tổng hợp chạy EDA, huấn luyện baseline, huấn luyện mô hình cải tiến, vẽ biểu đồ và xuất mô hình JSON cho trình duyệt.
- `build_html.py`: Script tạo file `AI05_Demo.html` từ kết quả trong `outputs/AI05/`.
- `data/AI05/SMSSpamCollection`: Bộ dữ liệu gốc UCI SMS Spam Collection (5.574 tin nhắn).
- `outputs/AI05/`: Chứa mô hình đã lưu (`.joblib`), `metrics.json`, `charts.json` và `browser_model.json`.

## 🚀 Cách chạy lại từ đầu trên máy (Local)

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy huấn luyện, EDA và xuất kết quả
python build_all.py

# 4. Tạo lại file HTML demo
python build_html.py
```

Hoặc chạy nhanh baseline:

```bash
python train.py --data "data/AI05/SMSSpamCollection"
python demo.py --text "Congratulations you won a prize claim now"
```

## 📚 Nguồn dữ liệu & Lưu ý

- **Nguồn dữ liệu:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (CC BY 4.0).
- **Lưu ý:** Bộ dữ liệu sử dụng tiếng Anh; kết quả chỉ mang tính chất tham khảo học thuật, không khẳng định khả năng nhận diện lừa đảo tiếng Việt.
