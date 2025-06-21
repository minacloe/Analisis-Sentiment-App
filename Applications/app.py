import sys

if sys.stdout is None:
    import io
    sys.stdout = io.StringIO()
if sys.stderr is None:
    import io
    sys.stderr = io.StringIO()

import tkinter as tk
from io import BytesIO
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageTk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from wordcloud import STOPWORDS, WordCloud

class ProgressCallback(Callback):
    def __init__(self, progress_bar, progress_label, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        self.progress_bar["maximum"] = self.epochs
        self.progress_label.config(text=f"Training... Epoch 0/{self.epochs}")
        self.progress_bar.master.update_idletasks()

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        self.progress_label.config(
            text=f"Training... Epoch {epoch + 1}/{self.epochs} | "
            f"Acc: {acc:.2f} | Val Acc: {val_acc:.2f}"
        )
        self.progress_bar["value"] = epoch + 1
        self.progress_bar.master.update_idletasks()

    def on_train_end(self, logs=None):
        self.progress_label.config(text="Training Selesai!")
        self.progress_bar.master.update_idletasks()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis Sentimen GUI BiLSTM (Optimized)")
        self.root.geometry("1366x768")
        self.root.configure(bg="#f0f0f0")

        self.df = None
        self.current_image = None
        self.tokenizer = None
        self.le = None
        self.model = None
        self.history = None

        self.max_words = 15000
        self.max_len = 80
        self.embedding_dim = 128

        main_container = tk.Frame(root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_section = tk.Frame(main_container, bg="#f0f0f0")
        top_section.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        left_panel = tk.Frame(
            top_section, width=300, bg="#ffffff", relief=tk.RAISED, bd=2
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        header_frame = tk.Frame(left_panel, bg="#2c3e50", height=50)
        header_frame.pack(fill=tk.X)
        tk.Label(
            header_frame,
            text="Control Panel",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 14, "bold"),
        ).pack(expand=True)

        control_frame = tk.Frame(left_panel, bg="#ffffff")
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(
            control_frame,
            text="📁 Pilih File CSV",
            command=self.load_file,
            bg="#3498db",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            pady=8,
        )
        self.btn_load.pack(fill=tk.X, pady=(0, 10))

        self.lbl_file = tk.Label(
            control_frame,
            text="Belum ada file dipilih",
            wraplength=280,
            justify="center",
            bg="#ffffff",
            fg="#7f8c8d",
            font=("Arial", 9),
        )
        self.lbl_file.pack(pady=5)

        self.btn_process = tk.Button(
            control_frame,
            text="🚀 Latih Model BiLSTM",
            command=self.process_data,
            state=tk.DISABLED,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            pady=8,
        )
        self.btn_process.pack(fill=tk.X, pady=(10, 5))

        info_frame = tk.Frame(left_panel, bg="#ecf0f1", relief=tk.SUNKEN, bd=1)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(
            info_frame,
            text="Informasi Dataset",
            bg="#ecf0f1",
            font=("Arial", 11, "bold"),
        ).pack(pady=5)
        self.lbl_info = tk.Label(
            info_frame,
            text="",
            bg="#ecf0f1",
            fg="#2c3e50",
            font=("Arial", 10),
            justify="left",
            anchor="nw",
        )
        self.lbl_info.pack(pady=5, padx=10, fill="both", expand=True)

        right_panel = tk.Frame(top_section, bg="#ffffff", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", "10", "bold"))
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(preview_tab, text="📊 Dataset Preview")
        self.preview_tree = self.create_treeview(preview_tab)

        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="✅ Hasil Prediksi")
        self.results_tree = self.create_treeview(results_tab)
        self.notebook.tab(1, state="disabled")

        bottom_section = tk.Frame(
            main_container, height=350, bg="#ffffff", relief=tk.RAISED, bd=2
        )
        bottom_section.pack(fill=tk.BOTH, expand=False, pady=(5, 0))
        bottom_section.pack_propagate(False)

        viz_header = tk.Frame(bottom_section, bg="#8e44ad", height=40)
        viz_header.pack(fill=tk.X)
        tk.Label(
            viz_header,
            text="Area Visualisasi",
            bg="#8e44ad",
            fg="white",
            font=("Arial", 12, "bold"),
        ).pack(expand=True)

        canvas_frame = tk.Frame(bottom_section, bg="#ffffff")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg="white", relief=tk.SUNKEN, bd=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(main_container, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Label(
            btn_frame,
            text="Pilih Visualisasi:",
            bg="#f0f0f0",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(10, 5))

        buttons_config = [
            ("☁️ Word Cloud", self.show_wordcloud, "#e74c3c"),
            ("🥧 Pie Chart Sentimen", self.show_pie_chart, "#9b59b6"),
            ("🔢 Confusion Matrix", self.show_confusion_matrix, "#f39c12"),
            ("📜 Classification Report", self.show_classification_report, "#16a085"),
            ("📈 Riwayat Training", self.show_training_history, "#e67e22"),
        ]
        for text, command, color in buttons_config:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=command,
                bg=color,
                fg="white",
                font=("Arial", 9, "bold"),
                relief=tk.FLAT,
                padx=12,
                pady=5,
            )
            btn.pack(side=tk.LEFT, padx=5)

    def create_treeview(self, parent):
        tree_frame = tk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y = tk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
            selectmode="extended",
        )
        tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)
        return tree

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.df = self.df.dropna(subset=["text", "sentiment"])
            self.df["text"] = self.df["text"].astype(str)
            filename = file_path.split("/")[-1]
            self.lbl_file.config(text=f"✅ {filename}")
            self.show_dataset_preview()
            self.notebook.tab(1, state="disabled")
            self.notebook.select(0)
            self.canvas.delete("all")
            self.update_dataset_info()
            if "text" in self.df.columns and "sentiment" in self.df.columns:
                self.btn_process.config(state=tk.NORMAL)
            else:
                self.btn_process.config(state=tk.DISABLED)
                messagebox.showwarning(
                    "Warning", "Kolom 'text' dan 'sentiment' wajib ada."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuka file CSV:\n{e}")

    def update_dataset_info(self):
        if self.df is None:
            return
        info_text = f"Total Baris: {len(self.df)}\n"
        info_text += f"Total Kolom: {len(self.df.columns)}\n\n"
        if "sentiment" in self.df.columns:
            sentiment_counts = self.df["sentiment"].value_counts()
            info_text += "Distribusi Sentimen:\n"
            for sentiment, count in sentiment_counts.items():
                info_text += f"  - {sentiment}: {count}\n"
        self.lbl_info.config(text=info_text)

    def populate_treeview(self, tree, dataframe, limit=None):
        tree.delete(*tree.get_children())
        tree["columns"] = list(dataframe.columns)
        tree["show"] = "headings"
        for col in dataframe.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.W)
        df_to_show = dataframe.head(limit) if limit else dataframe
        for _, row in df_to_show.iterrows():
            values = [str(row[col])[:200] for col in dataframe.columns]
            tree.insert("", tk.END, values=values)

    def show_dataset_preview(self):
        if self.df is None:
            return
        self.populate_treeview(self.preview_tree, self.df, limit=100)

    def show_results_dataset(self):
        if self.df is None:
            return
        cols = self.df.columns.tolist()
        if "pred_label" in cols and "sentiment" in cols:
            cols = ["text", "sentiment", "pred_label"] + [
                c for c in cols if c not in ["text", "sentiment", "pred_label"]
            ]
        self.populate_treeview(self.results_tree, self.df[cols])

    def build_model(self, num_classes):
        model = Sequential(
            [
                Embedding(
                    input_dim=self.max_words,
                    output_dim=self.embedding_dim,
                    input_length=self.max_len,
                    trainable=True,
                ),
                Bidirectional(
                    LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
                ),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def process_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Silakan pilih file CSV terlebih dahulu.")
            return
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Proses Training Model")
        progress_window.geometry("500x150")
        progress_window.resizable(False, False)
        progress_window.transient(self.root)
        progress_label = tk.Label(
            progress_window, text="Mempersiapkan data...", font=("Arial", 10)
        )
        progress_label.pack(pady=(20, 5))
        progress_bar = ttk.Progressbar(progress_window, mode="determinate", length=460)
        progress_bar.pack(pady=10, padx=20)
        self.root.update_idletasks()
        try:
            texts = self.df["text"].tolist()
            labels = self.df["sentiment"].tolist()
            self.le = LabelEncoder()
            y = self.le.fit_transform(labels)
            num_classes = len(self.le.classes_)
            y_cat = to_categorical(y, num_classes=num_classes)
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            X = pad_sequences(
                sequences, maxlen=self.max_len, padding="post", truncating="post"
            )
            class_weights = class_weight.compute_class_weight(
                "balanced", classes=np.unique(y), y=y
            )
            class_weight_dict = dict(enumerate(class_weights))
            self.model = self.build_model(num_classes)
            epochs = 15
            early_stop = EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
            )
            progress_callback = ProgressCallback(progress_bar, progress_label, epochs)
            self.history = self.model.fit(
                X,
                y_cat,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weight_dict,
                callbacks=[early_stop, reduce_lr, progress_callback],
                verbose=0,
            )
            progress_window.destroy()
            y_pred_prob = self.model.predict(X)
            y_pred = np.argmax(y_pred_prob, axis=1)
            pred_labels = self.le.inverse_transform(y_pred)
            if "pred_label" in self.df.columns:
                self.df = self.df.drop(columns=["pred_label"])
            self.df["pred_label"] = pred_labels
            self.show_results_dataset()
            self.notebook.tab(1, state="normal")
            self.notebook.select(1)
            accuracy = accuracy_score(labels, pred_labels)
            f1 = f1_score(labels, pred_labels, average="weighted", zero_division=0)
            msg = (
                f"✅ Training Selesai!\n\n"
                f"Akurasi: {accuracy:.4f}\n"
                f"F1-Score (Weighted): {f1:.4f}\n\n"
                "Hasil prediksi lengkap tersedia di tab 'Hasil Prediksi'."
            )
            messagebox.showinfo("Hasil Proses", msg)
        except Exception as e:
            if progress_window.winfo_exists():
                progress_window.destroy()
            messagebox.showerror("Error", f"Terjadi kesalahan saat training:\n{e}")
            import traceback
            traceback.print_exc()

    def show_wordcloud(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Silakan pilih data terlebih dahulu.")
            return
        text = " ".join(self.df["text"].dropna().astype(str).tolist())
        stopwords_id = {
            "dan", "yang", "di", "ke", "dari", "untuk", "pada", "adalah", "ini", "itu",
            "atau", "sebagai", "dengan", "oleh", "karena", "saat", "juga", "akan",
            "tetapi", "maka", "dalam", "lagi", "seperti", "agar", "bahwa",
            "indomaret", "alfamart", "https", "www", "com", "id", "co", "net", "yg", "ada", "saya", "kamu",
            "kita", "mereka", "dia", "semua", "bisa", "tidak", "ya", "sudah", "belum", "t", "aku", "aja",
            "gak", "nggak", "ngga", "gitu", "gini", "mau", "apa", "ga", "gue", "lu", "lo", "kalo", "kalau",
            "kayak", "kayaknya", "kaya", "sih", "dong", "nih", "tapi", "tak", "tuh", "gw", "bgt",
            "kak", "selamat", "periode", "bulan", "tahun", "hari", "minggu", "waktu", "sekarang",
            "kemarin", "besok", "lusa", "jam", "detik", "menit", "sebelumnya", "kemudian", "selanjutnya", "mei",
            "sama", "jadi", "buat", "nya", "udah", "lah", "dulu", "malah", "cuma", "tadi", "bener", "semoga",
            "kok", "doang", "emang", "kenapa", "coba", "minta", "tau", "liat", "kan", "jd", "punya", "enak",
            "top", "langsung", "ternyata", "pengen", "karna", "pake", "amp", "biar", "bikin", "deket", "keluar",
            "sampe", "tiap", "pas", "lewat", "point", "msh", "udh", "maaf", "lihat", "yaudah", "banget", "terus", "masih",
        }
        custom_stopwords = set(STOPWORDS).union(stopwords_id)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=custom_stopwords,
            collocations=False,
            max_words=150,
        ).generate(text)
        self.display_image(wordcloud.to_image())

    def show_pie_chart(self):
        if self.df is None or "sentiment" not in self.df.columns:
            messagebox.showwarning(
                "Warning", "Data belum dimuat atau kolom 'sentiment' tidak ada."
            )
            return
        counts = self.df["sentiment"].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values
        colors = plt.cm.Set2.colors[: len(labels)]
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"fontsize": 14},
        )
        ax.set_title("Distribusi Sentimen", fontsize=16, fontweight="bold")
        self.plot_to_canvas(plt)

    def show_confusion_matrix(self):
        if self.model is None or "pred_label" not in self.df.columns:
            messagebox.showwarning("Warning", "Silakan latih model terlebih dahulu.")
            return
        y_true = self.df["sentiment"]
        y_pred = self.df["pred_label"]
        labels = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="viridis",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 14},
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        self.plot_to_canvas(plt)

    def show_classification_report(self):
        if self.model is None or "pred_label" not in self.df.columns:
            messagebox.showwarning("Warning", "Silakan latih model terlebih dahulu.")
            return
        report = classification_report(
            self.df["sentiment"],
            self.df["pred_label"],
            output_dict=True,
            zero_division=0,
        )
        df_report = pd.DataFrame(report).transpose().round(3)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.set_title("Classification Report", fontsize=16, fontweight="bold", pad=20)
        tbl = ax.table(
            cellText=df_report.values,
            rowLabels=df_report.index,
            colLabels=df_report.columns,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.1, 1.8)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col == -1:
                cell.set_facecolor("#34495e")
                cell.set_text_props(weight="bold", color="white")
            if row > 0 and col > -1:
                cell.set_facecolor("#ecf0f1")
        self.plot_to_canvas(plt, bbox_inches="tight")

    def show_training_history(self):
        if self.history is None:
            messagebox.showwarning(
                "Warning",
                "Riwayat training tidak ditemukan. Latih model terlebih dahulu.",
            )
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Riwayat Training Model", fontsize=16, fontweight="bold")
        ax1.plot(
            self.history.history["accuracy"],
            "o-",
            label="Training Accuracy",
            color="#2ecc71",
        )
        ax1.plot(
            self.history.history["val_accuracy"],
            "o-",
            label="Validation Accuracy",
            color="#e74c3c",
        )
        ax1.set_title("Model Accuracy", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax2.plot(
            self.history.history["loss"], "o-", label="Training Loss", color="#3498db"
        )
        ax2.plot(
            self.history.history["val_loss"],
            "o-",
            label="Validation Loss",
            color="#f39c12",
        )
        ax2.set_title("Model Loss", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)
        self.plot_to_canvas(plt)

    def plot_to_canvas(self, plot, **savefig_kwargs):
        buf = BytesIO()
        plot.tight_layout()
        plot.savefig(buf, format="png", dpi=100, **savefig_kwargs)
        plot.close()
        buf.seek(0)
        img = Image.open(buf)
        self.display_image(img)

    def display_image(self, pil_image):
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        img_ratio = pil_image.width / pil_image.height
        canvas_ratio = canvas_width / canvas_height
        if img_ratio > canvas_ratio:
            new_width = int(canvas_width * 0.95)
            new_height = int(new_width / img_ratio)
        else:
            new_height = int(canvas_height * 0.95)
            new_width = int(new_height * img_ratio)
        img = pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.current_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.current_image)
        self.canvas.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
