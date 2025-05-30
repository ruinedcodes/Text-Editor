import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, colorchooser, font
import os
from tkinter import *
import json
from datetime import datetime
from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet
from collections import defaultdict, Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

class TextEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text Editor")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set theme colors
        self.bg_color = "#2E3440"
        self.text_bg = "#3B4252"
        self.text_fg = "#ECEFF4"
        self.accent_color = "#88C0D0"
        self.suggestion_bg = "#4C566A"
        self.suggestion_fg = "#ECEFF4"
        
        # Initialize variables
        self.current_file = None
        self.text_modified = False
        self.font_size = 12
        self.current_font = "Consolas"
        
        # Initialize spell checker and word suggestions
        self.spell = SpellChecker()
        self.word_frequency = defaultdict(int)
        self.load_word_frequency()
        
        # Initialize ML models
        self.initialize_ml_models()
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Create main menu
        self.create_menu()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main text area
        self.create_text_area()
        
        # Create suggestion box
        self.create_suggestion_box()
        
        # Create status bar
        self.create_status_bar()
        
        # Bind events
        self.bind_events()
        
        # Load settings
        self.load_settings()

    def initialize_ml_models(self):
        # Initialize n-gram models
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        
        # Initialize TF-IDF vectorizer for sentence similarity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.sentence_vectors = None
        self.sentences = []
        
        # Load existing models
        self.load_ml_models()

    def create_menu(self):
        menubar = Menu(self.root, bg=self.bg_color, fg=self.text_fg)
        
        # File Menu
        file_menu = Menu(menubar, tearoff=0, bg=self.bg_color, fg=self.text_fg)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_editor)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit Menu
        edit_menu = Menu(menubar, tearoff=0, bg=self.bg_color, fg=self.text_fg)
        edit_menu.add_command(label="Cut", command=lambda: self.text_area.event_generate("<<Cut>>"), accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=lambda: self.text_area.event_generate("<<Copy>>"), accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=lambda: self.text_area.event_generate("<<Paste>>"), accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Find", command=self.show_find_dialog, accelerator="Ctrl+F")
        edit_menu.add_command(label="Replace", command=self.show_replace_dialog, accelerator="Ctrl+H")
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View Menu
        view_menu = Menu(menubar, tearoff=0, bg=self.bg_color, fg=self.text_fg)
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Theme Menu
        theme_menu = Menu(menubar, tearoff=0, bg=self.bg_color, fg=self.text_fg)
        theme_menu.add_command(label="Nord", command=lambda: self.change_theme("nord"))
        theme_menu.add_command(label="Dark", command=lambda: self.change_theme("dark"))
        theme_menu.add_command(label="Light", command=lambda: self.change_theme("light"))
        menubar.add_cascade(label="Theme", menu=theme_menu)
        
        self.root.config(menu=menubar)

    def create_toolbar(self):
        toolbar = Frame(self.root, bg=self.bg_color)
        toolbar.pack(fill=X, padx=5, pady=2)
        
        # Font family combobox
        self.font_family = ttk.Combobox(toolbar, width=15, values=sorted(font.families()))
        self.font_family.set(self.current_font)
        self.font_family.pack(side=LEFT, padx=5)
        self.font_family.bind("<<ComboboxSelected>>", self.change_font)
        
        # Font size combobox
        self.font_size_combo = ttk.Combobox(toolbar, width=5, values=[8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 36, 48, 72])
        self.font_size_combo.set(self.font_size)
        self.font_size_combo.pack(side=LEFT, padx=5)
        self.font_size_combo.bind("<<ComboboxSelected>>", self.change_font_size)
        
        # Style buttons with modern look
        style = ttk.Style()
        style.configure('Toolbar.TButton', padding=5)
        
        # Bold button
        self.bold_btn = ttk.Button(toolbar, text="B", width=3, command=self.toggle_bold, style='Toolbar.TButton')
        self.bold_btn.pack(side=LEFT, padx=2)
        
        # Italic button
        self.italic_btn = ttk.Button(toolbar, text="I", width=3, command=self.toggle_italic, style='Toolbar.TButton')
        self.italic_btn.pack(side=LEFT, padx=2)
        
        # Underline button
        self.underline_btn = ttk.Button(toolbar, text="U", width=3, command=self.toggle_underline, style='Toolbar.TButton')
        self.underline_btn.pack(side=LEFT, padx=2)
        
        # Color button
        self.color_btn = ttk.Button(toolbar, text="Color", width=6, command=self.choose_color, style='Toolbar.TButton')
        self.color_btn.pack(side=LEFT, padx=5)
        
        # Spell check button
        self.spell_btn = ttk.Button(toolbar, text="Spell Check", width=10, command=self.check_spelling, style='Toolbar.TButton')
        self.spell_btn.pack(side=LEFT, padx=5)

    def create_text_area(self):
        # Create main frame
        self.main_frame = Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Create text area with scrollbar
        self.text_area = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            bg=self.text_bg,
            fg=self.text_fg,
            insertbackground=self.text_fg,
            selectbackground=self.accent_color,
            font=(self.current_font, self.font_size),
            padx=10,
            pady=10
        )
        self.text_area.pack(fill=BOTH, expand=True)
        
        # Configure tags for text styling
        self.text_area.tag_configure("bold", font=(self.current_font, self.font_size, "bold"))
        self.text_area.tag_configure("italic", font=(self.current_font, self.font_size, "italic"))
        self.text_area.tag_configure("underline", underline=1)
        self.text_area.tag_configure("misspelled", underline=True, underlinefg="red")

    def create_suggestion_box(self):
        # Create suggestion frame
        self.suggestion_frame = Frame(self.root, bg=self.suggestion_bg, height=100)
        self.suggestion_frame.pack(fill=X, padx=5, pady=2)
        
        # Create suggestion label
        self.suggestion_label = Label(
            self.suggestion_frame,
            text="Suggestions:",
            bg=self.suggestion_bg,
            fg=self.suggestion_fg,
            font=(self.current_font, 10)
        )
        self.suggestion_label.pack(anchor=W, padx=5, pady=2)
        
        # Create suggestion buttons frame
        self.suggestion_buttons_frame = Frame(self.suggestion_frame, bg=self.suggestion_bg)
        self.suggestion_buttons_frame.pack(fill=X, padx=5, pady=2)
        
        # Create sentence suggestion frame
        self.sentence_suggestion_frame = Frame(self.root, bg=self.suggestion_bg, height=100)
        self.sentence_suggestion_frame.pack(fill=X, padx=5, pady=2)
        
        # Create sentence suggestion label
        self.sentence_suggestion_label = Label(
            self.sentence_suggestion_frame,
            text="Sentence Suggestions:",
            bg=self.suggestion_bg,
            fg=self.suggestion_fg,
            font=(self.current_font, 10)
        )
        self.sentence_suggestion_label.pack(anchor=W, padx=5, pady=2)
        
        # Create sentence suggestion buttons frame
        self.sentence_suggestion_buttons_frame = Frame(self.sentence_suggestion_frame, bg=self.suggestion_bg)
        self.sentence_suggestion_buttons_frame.pack(fill=X, padx=5, pady=2)
        
        # Initially hide suggestion boxes
        self.suggestion_frame.pack_forget()
        self.sentence_suggestion_frame.pack_forget()

    def create_status_bar(self):
        self.status_bar = Label(
            self.root,
            text="Ready",
            bd=1,
            relief=SUNKEN,
            anchor=W,
            bg=self.bg_color,
            fg=self.text_fg
        )
        self.status_bar.pack(side=BOTTOM, fill=X)

    def bind_events(self):
        self.text_area.bind("<<Modified>>", self.on_text_modified)
        self.text_area.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        self.root.bind("<space>", self.on_space_press)

    def new_file(self):
        if self.text_modified:
            if messagebox.askyesno("Unsaved Changes", "Do you want to save changes?"):
                self.save_file()
        self.text_area.delete(1.0, END)
        self.current_file = None
        self.text_modified = False
        self.update_title()

    def open_file(self):
        if self.text_modified:
            if messagebox.askyesno("Unsaved Changes", "Do you want to save changes?"):
                self.save_file()
        
        file_path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.text_area.delete(1.0, END)
                    self.text_area.insert(1.0, file.read())
                self.current_file = file_path
                self.text_modified = False
                self.update_title()
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")

    def save_file(self):
        if self.current_file:
            try:
                content = self.text_area.get(1.0, END)
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.text_modified = False
                self.update_title()
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
        else:
            self.save_as()

    def save_as(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.current_file = file_path
            self.save_file()

    def exit_editor(self):
        if self.text_modified:
            if messagebox.askyesno("Unsaved Changes", "Do you want to save changes?"):
                self.save_file()
        self.root.quit()

    def show_find_dialog(self):
        find_dialog = Toplevel(self.root)
        find_dialog.title("Find")
        find_dialog.geometry("300x100")
        find_dialog.transient(self.root)
        
        Label(find_dialog, text="Find:").pack(pady=5)
        find_entry = Entry(find_dialog, width=30)
        find_entry.pack(pady=5)
        
        def find_text():
            search_term = find_entry.get()
            if search_term:
                start_pos = self.text_area.search(search_term, "1.0", END)
                if start_pos:
                    end_pos = f"{start_pos}+{len(search_term)}c"
                    self.text_area.tag_remove("search", "1.0", END)
                    self.text_area.tag_add("search", start_pos, end_pos)
                    self.text_area.tag_config("search", background="yellow", foreground="black")
                    self.text_area.see(start_pos)
                else:
                    messagebox.showinfo("Find", "Text not found")
        
        Button(find_dialog, text="Find", command=find_text).pack(pady=5)

    def show_replace_dialog(self):
        replace_dialog = Toplevel(self.root)
        replace_dialog.title("Replace")
        replace_dialog.geometry("300x150")
        replace_dialog.transient(self.root)
        
        Label(replace_dialog, text="Find:").pack(pady=5)
        find_entry = Entry(replace_dialog, width=30)
        find_entry.pack(pady=5)
        
        Label(replace_dialog, text="Replace with:").pack(pady=5)
        replace_entry = Entry(replace_dialog, width=30)
        replace_entry.pack(pady=5)
        
        def replace_text():
            search_term = find_entry.get()
            replace_term = replace_entry.get()
            if search_term:
                content = self.text_area.get(1.0, END)
                new_content = content.replace(search_term, replace_term)
                self.text_area.delete(1.0, END)
                self.text_area.insert(1.0, new_content)
                self.text_modified = True
                self.update_title()
        
        Button(replace_dialog, text="Replace", command=replace_text).pack(pady=5)

    def zoom_in(self):
        self.font_size += 2
        self.text_area.configure(font=(self.current_font, self.font_size))
        self.font_size_combo.set(self.font_size)

    def zoom_out(self):
        if self.font_size > 2:
            self.font_size -= 2
            self.text_area.configure(font=(self.current_font, self.font_size))
            self.font_size_combo.set(self.font_size)

    def reset_zoom(self):
        self.font_size = 12
        self.text_area.configure(font=(self.current_font, self.font_size))
        self.font_size_combo.set(self.font_size)

    def change_theme(self, theme):
        if theme == "nord":
            self.bg_color = "#2E3440"
            self.text_bg = "#3B4252"
            self.text_fg = "#ECEFF4"
            self.accent_color = "#88C0D0"
        elif theme == "dark":
            self.bg_color = "#1E1E1E"
            self.text_bg = "#252526"
            self.text_fg = "#D4D4D4"
            self.accent_color = "#007ACC"
        else:  # light
            self.bg_color = "#F5F5F5"
            self.text_bg = "#FFFFFF"
            self.text_fg = "#000000"
            self.accent_color = "#007ACC"
        
        self.apply_theme()

    def apply_theme(self):
        self.root.configure(bg=self.bg_color)
        self.text_area.configure(bg=self.text_bg, fg=self.text_fg)
        self.status_bar.configure(bg=self.bg_color, fg=self.text_fg)
        self.toolbar.configure(bg=self.bg_color)
        self.main_frame.configure(bg=self.bg_color)

    def change_font(self, event=None):
        self.current_font = self.font_family.get()
        self.text_area.configure(font=(self.current_font, self.font_size))

    def change_font_size(self, event=None):
        self.font_size = int(self.font_size_combo.get())
        self.text_area.configure(font=(self.current_font, self.font_size))

    def toggle_bold(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "bold" in current_tags:
                self.text_area.tag_remove("bold", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("bold", "sel.first", "sel.last")
        except:
            pass

    def toggle_italic(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "italic" in current_tags:
                self.text_area.tag_remove("italic", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("italic", "sel.first", "sel.last")
        except:
            pass

    def toggle_underline(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "underline" in current_tags:
                self.text_area.tag_remove("underline", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("underline", "sel.first", "sel.last")
        except:
            pass

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            try:
                self.text_area.tag_add("color", "sel.first", "sel.last")
                self.text_area.tag_config("color", foreground=color)
            except:
                pass

    def on_text_modified(self, event=None):
        self.text_modified = True
        self.update_title()
        self.text_area.edit_modified(False)
        
        # Update ML models with new text
        if self.text_modified:
            text = self.text_area.get("1.0", END)
            self.update_ml_models(text)

    def update_title(self):
        title = "Advanced Text Editor"
        if self.current_file:
            title = f"{os.path.basename(self.current_file)} - {title}"
        if self.text_modified:
            title = f"* {title}"
        self.root.title(title)

    def update_status_bar(self, event=None):
        try:
            line = self.text_area.index("insert").split('.')[0]
            col = self.text_area.index("insert").split('.')[1]
            self.status_bar.config(text=f"Line: {line}, Column: {col}")
        except:
            pass

    def load_settings(self):
        try:
            with open("editor_settings.json", "r") as f:
                settings = json.load(f)
                self.font_size = settings.get("font_size", 12)
                self.current_font = settings.get("font_family", "Consolas")
                self.change_theme(settings.get("theme", "nord"))
        except:
            pass

    def save_settings(self):
        settings = {
            "font_size": self.font_size,
            "font_family": self.current_font,
            "theme": "nord"  # Default theme
        }
        try:
            with open("editor_settings.json", "w") as f:
                json.dump(settings, f)
        except:
            pass

    def on_key_release(self, event):
        self.update_status_bar()
        if event.char.isalpha():
            self.show_word_suggestions()
            self.show_sentence_suggestions()

    def on_space_press(self, event):
        # Update word frequency when space is pressed
        current_word = self.get_current_word()
        if current_word:
            self.word_frequency[current_word.lower()] += 1
            self.save_word_frequency()

    def get_current_word(self):
        try:
            # Get the current line and column
            line = self.text_area.index("insert").split('.')[0]
            col = int(self.text_area.index("insert").split('.')[1])
            
            # Get the current line text
            line_text = self.text_area.get(f"{line}.0", f"{line}.end")
            
            # Find the current word
            words = line_text.split()
            current_pos = 0
            for word in words:
                word_start = line_text.find(word, current_pos)
                word_end = word_start + len(word)
                if word_start <= col <= word_end:
                    return word
                current_pos = word_end
        except:
            pass
        return None

    def show_word_suggestions(self):
        current_word = self.get_current_word()
        if not current_word:
            self.suggestion_frame.pack_forget()
            return

        # Get suggestions
        suggestions = self.get_suggestions(current_word)
        if not suggestions:
            self.suggestion_frame.pack_forget()
            return

        # Clear existing suggestion buttons
        for widget in self.suggestion_buttons_frame.winfo_children():
            widget.destroy()

        # Create suggestion buttons
        for suggestion in suggestions[:5]:  # Show top 5 suggestions
            btn = ttk.Button(
                self.suggestion_buttons_frame,
                text=suggestion,
                command=lambda s=suggestion: self.apply_suggestion(s),
                style='Toolbar.TButton'
            )
            btn.pack(side=LEFT, padx=2)

        # Show suggestion box
        self.suggestion_frame.pack(fill=X, padx=5, pady=2)

    def get_suggestions(self, word):
        suggestions = set()
        
        # Add spell checker suggestions
        if word.lower() not in self.spell:
            suggestions.update(self.spell.candidates(word))
        
        # Add wordnet synonyms
        for syn in wordnet.synsets(word):
            suggestions.update([lemma.name() for lemma in syn.lemmas()])
        
        # Add frequency-based suggestions
        word_lower = word.lower()
        similar_words = [w for w in self.word_frequency.keys() 
                        if w.startswith(word_lower) or 
                        self.levenshtein_distance(word_lower, w) <= 2]
        suggestions.update(sorted(similar_words, 
                                key=lambda x: self.word_frequency[x], 
                                reverse=True))
        
        return list(suggestions)

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def apply_suggestion(self, suggestion):
        try:
            # Get the current word position
            line = self.text_area.index("insert").split('.')[0]
            col = int(self.text_area.index("insert").split('.')[1])
            line_text = self.text_area.get(f"{line}.0", f"{line}.end")
            
            # Find the current word
            words = line_text.split()
            current_pos = 0
            for word in words:
                word_start = line_text.find(word, current_pos)
                word_end = word_start + len(word)
                if word_start <= col <= word_end:
                    # Replace the word
                    self.text_area.delete(f"{line}.{word_start}", f"{line}.{word_end}")
                    self.text_area.insert(f"{line}.{word_start}", suggestion)
                    break
                current_pos = word_end
            
            # Hide suggestion box
            self.suggestion_frame.pack_forget()
        except:
            pass

    def check_spelling(self):
        # Clear existing misspelled tags
        self.text_area.tag_remove("misspelled", "1.0", END)
        
        # Get all text
        text = self.text_area.get("1.0", END)
        words = re.findall(r'\b\w+\b', text)
        
        # Check each word
        for word in words:
            if word.lower() not in self.spell:
                # Find all occurrences of the word
                start_pos = "1.0"
                while True:
                    start_pos = self.text_area.search(r'\y' + word + r'\y', start_pos, END, regexp=True)
                    if not start_pos:
                        break
                    end_pos = f"{start_pos}+{len(word)}c"
                    self.text_area.tag_add("misspelled", start_pos, end_pos)
                    start_pos = end_pos

    def load_word_frequency(self):
        try:
            with open("word_frequency.json", "r") as f:
                self.word_frequency = defaultdict(int, json.load(f))
        except:
            pass

    def save_word_frequency(self):
        try:
            with open("word_frequency.json", "w") as f:
                json.dump(dict(self.word_frequency), f)
        except:
            pass

    def show_sentence_suggestions(self):
        current_sentence = self.get_current_sentence()
        if not current_sentence:
            self.sentence_suggestion_frame.pack_forget()
            return

        # Get sentence suggestions
        suggestions = self.get_sentence_suggestions(current_sentence)
        if not suggestions:
            self.sentence_suggestion_frame.pack_forget()
            return

        # Clear existing suggestion buttons
        for widget in self.sentence_suggestion_buttons_frame.winfo_children():
            widget.destroy()

        # Create suggestion buttons
        for suggestion in suggestions[:3]:  # Show top 3 suggestions
            btn = ttk.Button(
                self.sentence_suggestion_buttons_frame,
                text=suggestion,
                command=lambda s=suggestion: self.apply_sentence_suggestion(s),
                style='Toolbar.TButton'
            )
            btn.pack(side=LEFT, padx=2)

        # Show suggestion box
        self.sentence_suggestion_frame.pack(fill=X, padx=5, pady=2)

    def get_current_sentence(self):
        try:
            # Get the current line and column
            line = self.text_area.index("insert").split('.')[0]
            col = int(self.text_area.index("insert").split('.')[1])
            
            # Get the current line text
            line_text = self.text_area.get(f"{line}.0", f"{line}.end")
            
            # Find the current sentence
            sentences = sent_tokenize(line_text)
            current_pos = 0
            for sentence in sentences:
                sentence_start = line_text.find(sentence, current_pos)
                sentence_end = sentence_start + len(sentence)
                if sentence_start <= col <= sentence_end:
                    return sentence.strip()
                current_pos = sentence_end
        except:
            pass
        return None

    def get_sentence_suggestions(self, current_sentence):
        suggestions = set()
        
        # Get n-gram based suggestions
        words = word_tokenize(current_sentence.lower())
        if len(words) >= 2:
            # Get bigram suggestions
            last_word = words[-1]
            if last_word in self.bigrams:
                suggestions.update(self.bigrams[last_word].most_common(3))
        
        if len(words) >= 3:
            # Get trigram suggestions
            last_two_words = ' '.join(words[-2:])
            if last_two_words in self.trigrams:
                suggestions.update(self.trigrams[last_two_words].most_common(3))
        
        # Get similar sentences using TF-IDF and cosine similarity
        if self.sentence_vectors is not None and len(self.sentences) > 0:
            current_vector = self.vectorizer.transform([current_sentence])
            similarities = cosine_similarity(current_vector, self.sentence_vectors).flatten()
            similar_indices = similarities.argsort()[-3:][::-1]
            suggestions.update([self.sentences[i] for i in similar_indices])
        
        return list(suggestions)

    def apply_sentence_suggestion(self, suggestion):
        try:
            # Get the current sentence position
            line = self.text_area.index("insert").split('.')[0]
            col = int(self.text_area.index("insert").split('.')[1])
            line_text = self.text_area.get(f"{line}.0", f"{line}.end")
            
            # Find the current sentence
            sentences = sent_tokenize(line_text)
            current_pos = 0
            for sentence in sentences:
                sentence_start = line_text.find(sentence, current_pos)
                sentence_end = sentence_start + len(sentence)
                if sentence_start <= col <= sentence_end:
                    # Replace the sentence
                    self.text_area.delete(f"{line}.{sentence_start}", f"{line}.{sentence_end}")
                    self.text_area.insert(f"{line}.{sentence_start}", suggestion)
                    break
                current_pos = sentence_end
            
            # Hide suggestion box
            self.sentence_suggestion_frame.pack_forget()
        except:
            pass

    def update_ml_models(self, text):
        # Update n-gram models
        words = word_tokenize(text.lower())
        
        # Update bigrams
        for bigram in ngrams(words, 2):
            self.bigrams[bigram[0]][bigram[1]] += 1
        
        # Update trigrams
        for trigram in ngrams(words, 3):
            self.trigrams[' '.join(trigram[:2])][trigram[2]] += 1
        
        # Update sentence model
        sentences = sent_tokenize(text)
        self.sentences.extend(sentences)
        if len(self.sentences) > 1000:  # Keep only last 1000 sentences
            self.sentences = self.sentences[-1000:]
        
        # Update TF-IDF vectors
        if len(self.sentences) > 0:
            self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)
        
        # Save updated models
        self.save_ml_models()

    def load_ml_models(self):
        try:
            with open("ml_models.json", "r") as f:
                data = json.load(f)
                self.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in data['bigrams'].items()})
                self.trigrams = defaultdict(Counter, {k: Counter(v) for k, v in data['trigrams'].items()})
                self.sentences = data['sentences']
                if len(self.sentences) > 0:
                    self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)
        except:
            pass

    def save_ml_models(self):
        try:
            data = {
                'bigrams': {k: dict(v) for k, v in self.bigrams.items()},
                'trigrams': {k: dict(v) for k, v in self.trigrams.items()},
                'sentences': self.sentences
            }
            with open("ml_models.json", "w") as f:
                json.dump(data, f)
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    editor = TextEditor(root)
    root.mainloop() 