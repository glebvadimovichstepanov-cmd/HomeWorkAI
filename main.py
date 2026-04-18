"""
🎓 Локальный ИИ-репетитор с мультимодальным распознаванием (llama.cpp + mmproj)
Работает с llama-server --mmproj через OpenAI-compatible API
Поддерживает: загрузку фото, вставку из буфера (Ctrl+V), streaming-ответы, SymPy-проверку
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab
import requests
import json
import os
import re
import base64
import io
import threading
import sympy
from sympy.parsing.latex import parse_latex
from datetime import datetime

# ==========================================
# ⚙️ КОНФИГУРАЦИЯ
# ==========================================
LLM_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
HISTORY_FILE = "homework_history.json"
MAX_HISTORY = 20
MAX_IMAGE_SIZE = (1024, 1024)  # Оптимизация: уменьшаем большие изображения


class HomeworkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎓 ИИ-репетитор | Мультимодальный OCR")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.current_image: Image.Image = None
        self.history = self._load_history()

        self._check_server()
        self._create_ui()
        self._bind_shortcuts()
        self._log_status("✅ Готово. Вставьте фото (Ctrl+V) или загрузите файл")

    # ==========================================
    # 🔍 ПРОВЕРКА СЕРВЕРА
    # ==========================================
    def _check_server(self):
        try:
            requests.get("http://127.0.0.1:8080/health", timeout=3)
        except Exception:
            messagebox.showwarning("Сервер не запущен",
                "llama-server.exe не отвечает на http://127.0.0.1:8080\n"
                "Запустите сервер с флагом --mmproj перед использованием.")
            self.quit()

    # ==========================================
    # 🖼️ ИНТЕРФЕЙС
    # ==========================================
    def _create_ui(self):
        # Левая панель
        left = ctk.CTkFrame(self, width=260)
        left.pack(side="left", fill="y", padx=10, pady=10)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="📚 Домашние задания", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 15))

        self.btn_load = ctk.CTkButton(left, text="📷 Загрузить фото", command=self.load_image, height=40)
        self.btn_load.pack(pady=5, padx=20, fill="x")

        self.btn_clipboard = ctk.CTkButton(left, text="📋 Вставить из буфера", command=self.paste_from_clipboard, height=40, fg_color="#2196F3")
        self.btn_clipboard.pack(pady=5, padx=20, fill="x")

        self.btn_clear = ctk.CTkButton(left, text="🗑️ Очистить", command=self.clear_all, fg_color="gray", hover_color="darkgray")
        self.btn_clear.pack(pady=5, padx=20, fill="x")

        ctk.CTkFrame(left, height=2, fg_color="gray").pack(pady=15, padx=20, fill="x")

        ctk.CTkLabel(left, text="Предмет:", font=ctk.CTkFont(size=14)).pack(pady=(10, 5), padx=20, anchor="w")
        self.subject_var = ctk.StringVar(value="математика")
        ctk.CTkOptionMenu(left, variable=self.subject_var,
            values=["математика", "физика", "химия", "биология", "русский язык", "литература",
                   "история", "обществознание", "английский", "геометрия", "алгебра"],
            font=ctk.CTkFont(size=13)).pack(pady=5, padx=20, fill="x")

        ctk.CTkFrame(left, height=2, fg_color="gray").pack(pady=15, padx=20, fill="x")

        ctk.CTkLabel(left, text="📋 История:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10, padx=20, anchor="w")
        self.history_frame = ctk.CTkScrollableFrame(left, height=350)
        self.history_frame.pack(pady=5, padx=10, fill="both", expand=True)
        self._update_history_ui()

        # Правая панель
        right = ctk.CTkFrame(self)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Предпросмотр изображения
        self.img_frame = ctk.CTkFrame(right, height=200)
        self.img_frame.pack(fill="x", pady=(0, 10))
        self.lbl_image = ctk.CTkLabel(self.img_frame, text="📷 Здесь будет фото задачи", font=ctk.CTkFont(size=16), justify="center")
        self.lbl_image.pack(expand=True, fill="both", padx=20, pady=15)

        # Поле для дополнительных уточнений (опционально)
        ctk.CTkLabel(right, text="💬 Дополнительные указания (необязательно):", font=ctk.CTkFont(size=14)).pack(pady=(5, 5), padx=10, anchor="w")
        self.txt_hints = ctk.CTkTextbox(right, height=60, font=ctk.CTkFont(size=13), wrap="word")
        self.txt_hints.pack(pady=5, padx=10, fill="x")
        self.txt_hints.insert("1.0", "Реши пошагово, объясни каждый шаг")

        # Кнопки действий
        btn_row = ctk.CTkFrame(right)
        btn_row.pack(pady=10, padx=10, fill="x")

        self.btn_sympy = ctk.CTkButton(btn_row, text="🧮 SymPy проверка", command=self.run_sympy, fg_color="#4CAF50", height=35)
        self.btn_sympy.pack(side="left", padx=5, fill="x", expand=True)

        self.btn_solve = ctk.CTkButton(btn_row, text="🧠 Получить решение", command=self.solve_task, height=40, font=ctk.CTkFont(size=14, weight="bold"), fg_color="#2CC985")
        self.btn_solve.pack(side="left", padx=5, fill="x", expand=True)

        # Ответ
        ctk.CTkLabel(right, text="✅ Решение:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5), padx=10, anchor="w")
        self.txt_answer = ctk.CTkTextbox(right, font=ctk.CTkFont(size=13), wrap="word", state="disabled")
        self.txt_answer.pack(pady=5, padx=10, fill="both", expand=True)

        self.lbl_status = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=12), text_color="gray")
        self.lbl_status.pack(pady=5, padx=10)

    def _bind_shortcuts(self):
        """Привязка глобальных горячих клавиш"""
        self.bind("<Control-v>", lambda e: self.paste_from_clipboard())
        self.bind("<Control-V>", lambda e: self.paste_from_clipboard())

    # ==========================================
    # 🧵 УТИЛИТЫ
    # ==========================================
    def _log_status(self, msg): self.lbl_status.configure(text=msg)

    def _set_ui_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_load.configure(state=state)
        self.btn_clipboard.configure(state=state)
        self.btn_clear.configure(state=state)
        self.btn_sympy.configure(state=state)
        self.btn_solve.configure(state=state if self.current_image else "disabled")

    def _run_async(self, task, start_msg="Обработка...", success_msg="✅ Готово"):
        self._log_status(start_msg)
        self._set_ui_state(False)
        def worker():
            try:
                task()
                self.after(0, lambda: self._log_status(success_msg))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, lambda: self._log_status("❌ Ошибка"))
            finally:
                self.after(0, lambda: self._set_ui_state(True))
        threading.Thread(target=worker, daemon=True).start()

    def _image_to_base64(self, img: Image.Image) -> str:
        """Конвертирует PIL.Image в base64 строку для API"""
        # Оптимизация: уменьшаем изображение для ускорения передачи
        if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
            img = img.copy()
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Конвертируем в RGB (убираем альфа-канал для совместимости)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P': img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _show_image(self, img: Image.Image):
        """Отображает изображение в интерфейсе"""
        preview = img.copy()
        preview.thumbnail((400, 180), Image.Resampling.LANCZOS)
        self.photo_image = ctk.CTkImage(light_image=preview, dark_image=preview, size=(400, 180))
        self.lbl_image.configure(text="", image=self.photo_image)
        self.btn_solve.configure(state="normal")

    # ==========================================
    # 📷 ЗАГРУЗКА ИЗОБРАЖЕНИЙ
    # ==========================================
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path: return
        try:
            self.current_image = Image.open(path)
            self._show_image(self.current_image)
            self._log_status("✅ Фото загружено")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")

    def paste_from_clipboard(self):
        """Вставка изображения из буфера обмена (Ctrl+V)"""
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                # Пробуем альтернативный метод для некоторых систем
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                if root.clipboard_get():
                    root.destroy()
                    self._log_status("⚠️ В буфере текст, а не изображение")
                    return
                root.destroy()
                self._log_status("⚠️ Буфер обмена пуст или не содержит изображение")
                return

            self.current_image = img.convert("RGB") if img.mode != "RGB" else img
            self._show_image(self.current_image)
            self._log_status("✅ Изображение вставлено из буфера")
        except Exception as e:
            messagebox.showerror("Ошибка буфера", f"Не удалось вставить изображение:\n{e}")

    # ==========================================
    # 🧮 SYMPY ПРОВЕРКА (для формул из ответа)
    # ==========================================
    def run_sympy(self):
        def task():
            answer = self.txt_answer.get("1.0", "end").strip()
            if not answer: raise ValueError("Сначала получите решение")

            # Ищем формулы в формате $$...$$ или $...$
            blocks = re.findall(r'\$\$?\s*([^$]+?)\s*\$\$?', answer, re.DOTALL)
            if not blocks: raise ValueError("Не найдено формул для проверки")

            report = ["🧮 Проверка вычислений (SymPy):"]
            for b in blocks:
                b = b.strip()
                try:
                    if '=' in b:
                        l, r = b.split('=', 1)
                        eq = parse_latex(l.strip()) - parse_latex(r.strip())
                        sols = sympy.solve(eq)
                        report.append(f"📐 ${b}$\n✅ Корни: {sols}")
                    else:
                        expr = parse_latex(b)
                        simp = sympy.simplify(expr)
                        val = expr.evalf()
                        report.append(f"📐 ${b}$\n🔄 Упрощение: ${sympy.latex(simp)}$\n🔢 ≈ {float(val):.4f}")
                except Exception as e:
                    report.append(f"⚠️ Ошибка: {e}")
            self.after(0, lambda: self.txt_answer.insert("end", "\n\n" + "\n\n".join(report)))
        self._run_async(task, "⏳ Проверка...", "✅ Проверка завершена")

    # ==========================================
    # 🧠 ЗАПРОС К МУЛЬТИМОДАЛЬНОЙ МОДЕЛИ
    # ==========================================
    def solve_task(self):
        if not self.current_image:
            messagebox.showwarning("Внимание", "Загрузите или вставьте изображение задачи!")
            return

        subject = self.subject_var.get()
        hints = self.txt_hints.get("1.0", "end").strip()

        self.txt_answer.configure(state="normal")
        self.txt_answer.delete("1.0", "end")
        self._log_status("⏳ Анализ изображения и генерация решения...")
        self._set_ui_state(False)

        # Формируем промпт для мультимодальной модели
        system_prompt = """Ты — школьный ИИ-репетитор. Тебе присылают фото задач из учебников.
ТВОИ ЗАДАЧИ:
1. Распознай текст и формулы на изображении (включая рукописные заметки)
2. Реши задачу пошагово с подробными объяснениями
3. Указывай используемые формулы, теоремы, правила
4. Пиши простым языком для ученика 7-11 класса
5. В конце делай вывод и указывай на типичные ошибки
6. Используй LaTeX для формул: $$ ... $$
НИКОГДА не давай только ответ без объяснений."""

        user_content = [
            {"type": "text", "text": f"Предмет: {subject}\n{hints}\n\nРаспознай и реши задачу на изображении:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._image_to_base64(self.current_image)}"}}
        ]

        def task():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            # Streaming-ответ
            for chunk in self._stream_llm(messages, temp=0.3, top_p=0.8, top_k=20):
                self.after(0, lambda c=chunk: self.txt_answer.insert("end", c))

            # Сохраняем в историю
            full_answer = self.txt_answer.get("1.0", "end").strip()
            self._save_history(subject, "[Изображение]", full_answer)
            self.after(0, self._update_history_ui)
            self.after(0, lambda: self.txt_answer.configure(state="disabled"))
            self.after(0, lambda: self._log_status("✅ Решение готово"))
            self.after(0, lambda: self._set_ui_state(True))

        threading.Thread(target=task, daemon=True).start()

    def _stream_llm(self, messages, temp=0.3, top_p=0.9, top_k=40):
        """Streaming-запрос к llama-server с поддержкой изображений"""
        payload = {
            "model": "local",  # Игнорируется сервером, но требуется валидацией API
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
            "max_tokens": 4096
        }

        try:
            with requests.post(LLM_API_URL, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str == "[DONE]": break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content: yield content
                            except json.JSONDecodeError: continue
        except requests.exceptions.ConnectionError:
            yield "\n❌ Не удалось подключиться к llama-server. Убедитесь, что он запущен."
        except Exception as e:
            yield f"\n❌ Ошибка: {e}"

    # ==========================================
    # 📂 ИСТОРИЯ
    # ==========================================
    def _save_history(self, subject, question, answer):
        item = {
            "date": datetime.now().strftime("%d.%m %H:%M"),
            "subject": subject,
            "question": question,
            "answer_preview": answer[:200] + ("..." if len(answer) > 200 else "")
        }
        self.history.insert(0, item)
        if len(self.history) > MAX_HISTORY: self.history = self.history[:MAX_HISTORY]
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except: pass

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return []

    def _update_history_ui(self):
        for w in self.history_frame.winfo_children(): w.destroy()
        for item in self.history[:7]:
            f = ctk.CTkFrame(self.history_frame)
            f.pack(fill="x", pady=3, padx=5)
            ctk.CTkLabel(f, text=f"{item['date']} • {item['subject']}",
                         font=ctk.CTkFont(size=10, weight="bold")).pack(padx=8, pady=(3,0), anchor="w")
            ctk.CTkLabel(f, text=item['answer_preview'],
                         font=ctk.CTkFont(size=9), wraplength=220, justify="left").pack(padx=8, pady=(0,3), anchor="w")

    def clear_all(self):
        self.current_image = None
        self.lbl_image.configure(text="📷 Здесь будет фото задачи", image="")
        self.txt_hints.delete("1.0", "end")
        self.txt_hints.insert("1.0", "Реши пошагово, объясни каждый шаг")
        self.txt_answer.configure(state="normal")
        self.txt_answer.delete("1.0", "end")
        self.txt_answer.configure(state="disabled")
        self.btn_solve.configure(state="disabled")
        self._log_status("")

if __name__ == "__main__":
    app = HomeworkApp()
    app.mainloop()