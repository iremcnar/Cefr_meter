

import tkinter as tk
from tkinter import messagebox
import joblib

# Gerekli modelleri yükle
model = joblib.load('cefr_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# Tahmin fonksiyonu
def predict_cefr_level(text):
    processed = text.lower()
    vector = tfidf.transform([processed])
    prediction = model.predict_proba(vector)[0]
    predicted_index = prediction.argmax()
    predicted_level = le.inverse_transform([predicted_index])[0]
    top2 = sorted(zip(le.classes_, prediction), key=lambda x: x[1], reverse=True)[:2]
    return predicted_level, top2

# GUI oluştur
def on_predict():
    sentence = entry.get()
    if not sentence.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir cümle girin.")
        return
    level, probs = predict_cefr_level(sentence)
    result_var.set(f"Tahmin Edilen Seviye: {level}\n\nEn Yüksek 2 Olasılık:\n" +
                   "\n".join([f"{lvl}: %{prob*100:.2f}" for lvl, prob in probs]))

# Pencere
root = tk.Tk()
root.title("CEFR Seviye Tahmini")

tk.Label(root, text="İngilizce bir cümle yazın:").pack(pady=5)
entry = tk.Entry(root, width=80)
entry.pack(padx=10, pady=5)

tk.Button(root, text="Tahmin Et", command=on_predict).pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, justify="left")
result_label.pack(pady=10)

root.mainloop()

