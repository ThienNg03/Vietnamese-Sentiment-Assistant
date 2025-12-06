

import tkinter as tk
from tkinter import ttk, messagebox
from transformers import pipeline
import sqlite3
from datetime import datetime
import re
import torch
import threading
from underthesea import word_tokenize
import unicodedata

MODEL_NAME = "distilbert-base-multilingual-cased"
THRESHOLD = 0.5
DB_PATH = "sentiments.db"

# -------- DB helpers  --------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sentiments (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT NOT NULL,
                   sentiment TEXT NOT NULL,
                   score REAL,
                   timestamp TEXT
                   )""")
    conn.commit()
    return conn

conn = init_db()

def insert_record(text, sentiment, score):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("INSERT INTO sentiments (text, sentiment, score, timestamp) VALUES (?, ?, ?, ?)",
                (text, sentiment, score, ts))
    conn.commit()

def fetch_recent(limit=50):
    cur = conn.cursor()
    cur.execute("SELECT timestamp, text, sentiment, score FROM sentiments ORDER BY timestamp DESC LIMIT ?", (limit,))
    return cur.fetchall()

# -------- Preprocessing  --------
# Từ điển cơ bản - chỉ các từ phổ biến (10-20 từ)
ABBR_MAP = {
    "rat": "rất",
    "r": "rất",
    "ko": "không",
    "k": "không",
    "hok": "không",
    "thik": "thích",
    "vs": "với",
    "dc": "được",
    "ok": "tốt",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "like": "thích",
    "love": "yêu",
    "bad": "xấu",
    "good": "tốt",
    "great": "tuyệt vời"
}

EMOJI_MAP = {
    ":)": "vui vẻ", ":d": "vui vẻ", ":p": "vui vẻ",
    ":(": "buồn", ":<": "buồn",
    "❤️": "yêu thích", "👍": "tốt", "😂": "rất vui"
}

def normalize_text(s: str) -> str:
    
    s = s or ""
    s = s.strip()
    
    if not s:
        return s
    
    # Bước 1: Unicode normalize (chuẩn hóa ký tự tổng hợp)
    s = unicodedata.normalize('NFC', s)
    
    # Bước 2: Chuyển sang chữ thường
    s = s.lower()
    
    # Bước 3: Thay thế emoji
    for emoji, text in EMOJI_MAP.items():
        s = s.replace(emoji, f" {text} ")

    # Bước 4: Chuẩn hóa ký tự lặp lại (ví dụ: tuyệtttt -> tuyệt).
    # Giữ lại tối đa 1 ký tự.
    s = re.sub(r'([a-záàảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳỵỷỹýđ])\1{2,}', r'\1', s)

    # Bước 5: Xóa ký tự đặc biệt nhưng giữ dấu câu cơ bản (? ! . , - ')
    s = re.sub(r"[^0-9a-záàảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳỵỷỹýđ\s?!.,\-']", " ", s)
    
    # Bước 6: Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    # Bước 7: Tách từ bằng underthesea
    try:
        tokens = word_tokenize(s)
    except Exception:
        # Fallback: split đơn giản
        tokens = s.split()
    
    # Bước 8: Thay thế từ viết tắt
    tokens = [ABBR_MAP.get(t, t) for t in tokens]
    
    # Bước 9: Xóa token rỗng
    tokens = [t.strip() for t in tokens if t.strip()]
    
    # Join lại
    s = " ".join(tokens)
    
    return s.strip()

# --------- Load model pipeline (Chỉ load 1 lần) ----------
pipe = None

def load_pipeline_with_fallback():
    global pipe
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        print(f"Đang tải model: {MODEL_NAME}")
        pipe = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
        print(f"✓ Tải thành công model: {MODEL_NAME}\n")
    except Exception as e:
        print(f"✗ Lỗi tải model '{MODEL_NAME}': {e}")
        raise RuntimeError(f"Không thể khởi tạo model {MODEL_NAME}")

def map_label(results: list):
    """
    Chuẩn hoá nhãn từ model.
    Xử lý cả trường hợp nhãn là "POSITIVE", "NEGATIVE" và "LABEL_0", "LABEL_1", "LABEL_2".
    """
    if not results:
        return "NEUTRAL", 0.0, {}
    
    # Sắp xếp kết quả theo score giảm dần
    sorted_results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
    best_result = sorted_results[0]
    best_label_raw = best_result.get("label", "").lower()
    best_score = best_result.get("score", 0.0)
    
    # Ưu tiên kiểm tra tên nhãn rõ ràng trước
    if "positive" in best_label_raw:
        return "POSITIVE", best_score
    elif "negative" in best_label_raw:
        return "NEGATIVE", best_score
    elif "neutral" in best_label_raw:
        return "NEUTRAL", best_score
    else:
        label_map = {"label_2": "POSITIVE", "label_1": "NEUTRAL", "label_0": "NEGATIVE"}
        return label_map.get(best_label_raw, "NEUTRAL"), best_score

def post_process_result(text: str, label: str, score: float, all_results: list):
    """
    Áp dụng 5 quy tắc sau xử lý để cải thiện kết quả theo tài liệu MODEL_BUILDING_PROCESS.md.
    """
    # --- Lấy thông tin cần thiết ---
    score_dict = {r.get("label").lower(): float(r.get("score", 0.0)) for r in all_results}
    
    # Chuẩn hóa các key có thể có của model (ví dụ: 'label_0', 'negative')
    pos_score = score_dict.get("positive", score_dict.get("label_2", 0.0))
    neg_score = score_dict.get("negative", score_dict.get("label_0", 0.0))
    neu_score = score_dict.get("neutral", score_dict.get("label_1", 0.0))

    scores_sorted = sorted([pos_score, neg_score, neu_score], reverse=True)
    confidence_gap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]

    final_label, final_score = label, score
    text_lower = text.lower()

    # --- Định nghĩa các bộ từ khóa ---
    negation_words = ["không", "ko", "chẳng", "chả", "chưa"]
    strong_negation_words = ["không hề", "chẳng hề", "không chút nào"] # Có thể gộp vào negation_words
    affirmation_words = ["rất", "quá", "cực", "vô cùng", "hết sức", "lắm", "thực sự", "ghê"]
    positive_keywords = ["tuyệt vời", "xuất sắc", "tốt", "thích", "ưng ý", "hài lòng", "đẹp", "chất lượng", "nhanh", "hoàn hảo", "tuyệt"]
    negative_keywords = ["tệ", "kém", "chán", "thất vọng", "dở", "xấu", "lâu", "chậm", "hỏng", "vỡ", "sai", "lỗi"]

    has_negation = any(word in text_lower for word in negation_words + strong_negation_words)
    has_affirmation = any(word in text_lower for word in affirmation_words)

    # --- Áp dụng các quy tắc tuần tự ---

    # Quy tắc 1: Phân tích khoảng cách tin cậy (Confidence Gap)
    if confidence_gap < 0.25: # Tăng ngưỡng để chắc chắn hơn
        final_label = "NEUTRAL"
        final_score = scores_sorted[0]

    # Quy tắc 2: Dựa trên từ khóa cảm xúc mạnh
    if any(word in text_lower for word in positive_keywords) and not has_negation:
        final_label = "POSITIVE"
        final_score = min(score + 0.2, 1.0) # Tăng độ tin cậy
    elif any(word in text_lower for word in negative_keywords) and not has_negation:
        final_label = "NEGATIVE"
        final_score = min(score + 0.2, 1.0)

    # Quy tắc 3: Xử lý phủ định (Negation)
    if has_negation:
        # Ví dụ: "không tệ", "chẳng xấu" -> NEUTRAL hoặc POSITIVE nhẹ
        if any(f"{neg_word} {bad_word}" in text_lower for neg_word in negation_words for bad_word in negative_keywords):
            final_label = "NEUTRAL"
            final_score = max(neu_score, 0.6)
        # Ví dụ: "không tốt", "không thích" -> NEGATIVE
        elif final_label == "POSITIVE":
            final_label = "NEGATIVE"
            final_score = min(score + 0.1, 1.0)

    # Quy tắc 4: Xử lý nhấn mạnh (Affirmation)
    if has_affirmation:
        if final_label in ["POSITIVE", "NEGATIVE"]:
            final_score = min(final_score + 0.1, 1.0)

    
    # Quy tắc 5: Xử lý câu hỏi
    if "?" in text and any(word in text_lower for word in negative_keywords):
        final_label = "NEGATIVE"
        final_score = min(score + 0.15, 1.0)

    # Quy tắc 6: Phạt điểm cho văn bản ngắn
    if len(text) < 15:
        final_score *= 0.9  # Giảm 10%

   
    # Quy tắc 7: Áp dụng ngưỡng cuối cùng
    if final_score < THRESHOLD:
        final_label = "NEUTRAL"

    return final_label, final_score

# --------- Tkinter UI Application ----------
class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Assistant")
        self.geometry("800x600")

        # --- UI Components ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Nhập câu tiếng Việt:", font=("Helvetica", 12)).pack(pady=(0, 5), anchor="w")

        # --- Khung chứa ô nhập liệu và nút bấm ---
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.text_input = tk.Text(input_frame, height=1, width=50, font=("Helvetica", 10))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.analyze_button = ttk.Button(input_frame, text="Phân loại", command=self.start_analysis_thread)
        self.analyze_button.pack(side=tk.LEFT)

        self.result_label = ttk.Label(main_frame, text="Kết quả: ", font=("Helvetica", 11, "bold"))
        self.result_label.pack(pady=10, anchor="w")

        ttk.Label(main_frame, text="Lịch sử phân loại :", font=("Helvetica", 12)).pack(pady=(10, 5), anchor="w")

        # --- History Table (Treeview) ---
        cols = ("Timestamp", "Text", "Sentiment", "Score")
        self.history_tree = ttk.Treeview(main_frame, columns=cols, show='headings')
        for col in cols:
            self.history_tree.heading(col, text=col)
        self.history_tree.column("Timestamp", width=140, anchor="w")
        self.history_tree.column("Text", width=400, anchor="w")
        self.history_tree.column("Sentiment", width=100, anchor="center")
        self.history_tree.column("Score", width=80, anchor="center")
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        # --- Cấu hình tag màu cho Treeview ---
        self.history_tree.tag_configure("POSITIVE", foreground="green")
        self.history_tree.tag_configure("NEGATIVE", foreground="red")
        self.history_tree.tag_configure("NEUTRAL", foreground="gray")

        # --- Delete Button ---
        self.delete_button = ttk.Button(main_frame, text="Xóa tất cả lịch sử", command=self.delete_all_records)
        self.delete_button.pack(pady=10)

        self.load_history()


    def load_history(self):
        # Clear existing items
        for i in self.history_tree.get_children():
            self.history_tree.delete(i)
        # Fetch and insert new items
        rows = fetch_recent(50)
        for row in rows:
            # row = (timestamp, text, sentiment, score)
            sentiment = row[2]
            # Format score to 2 decimal places for display
            formatted_row = list(row)
            formatted_row[3] = f"{row[3]:.2f}"
            self.history_tree.insert("", "end", values=formatted_row, tags=(sentiment,))

    def delete_all_records(self):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa tất cả lịch sử không?"):
            cur = conn.cursor()
            cur.execute("DELETE FROM sentiments")
            conn.commit()
            self.load_history()
            messagebox.showinfo("Hoàn tất", "Đã xóa tất cả lịch sử.")

    def start_analysis_thread(self):
        # Disable button to prevent multiple clicks
        self.analyze_button.config(state=tk.DISABLED)
        self.result_label.config(text="Đang phân tích...")
        # Run analysis in a separate thread to not freeze the UI
        analysis_thread = threading.Thread(target=self.analyze_sentiment)
        analysis_thread.start()

    def analyze_sentiment(self):
        user_input = self.text_input.get("1.0", tk.END)
        text = normalize_text(user_input)

        if len(text.strip()) < 5:
            self.update_ui_after_analysis("Vui lòng nhập tối thiểu 5 ký tự.", is_error=True)
            return

        # --- Sử dụng AI Model Pipeline ---
        try:
            # Gửi câu chuẩn hóa qua pipeline sentiment-analysis
            results = pipe(text, top_k=None)  # Lấy tất cả scores để chọn cao nhất
            
            if not results or len(results) == 0:
                self.update_ui_after_analysis("Không thể phân tích. Vui lòng thử lại.", is_error=True)
                return
            
            # Map label từ kết quả thô
            label, score = map_label(results)
            
            # --- POST-PROCESSING: Cải thiện kết quả ---
            label, score = post_process_result(text, label, score, results)
            
        except Exception as e:
            self.update_ui_after_analysis(f"Lỗi khi gọi model: {e}", is_error=True)
            return

        insert_record(text, label, score)
        result_text = f"Kết quả: {label} (score={score:.2f})"
        self.update_ui_after_analysis(result_text)
        self.load_history()

    def update_ui_after_analysis(self, message, is_error=False):
        if is_error:
            self.result_label.config(text=message, foreground="red")
        else:
            self.result_label.config(text=message, foreground="green")
        self.analyze_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    print("Đang tải model, vui lòng chờ...")
    load_pipeline_with_fallback()
    print("Tải model hoàn tất. Đang khởi động ứng dụng.")
    app = SentimentApp()
    app.mainloop()
