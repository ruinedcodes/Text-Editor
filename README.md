# Text Editor

A text editor built with Python and Tkinter, featuring machine learning capabilities for intelligent text suggestions.

## Features

### Core Features
- Modern GUI with multiple themes (Nord, Dark, Light)
- File operations (New, Open, Save, Save As)
- Text formatting (Bold, Italic, Underline, Color)
- Find and Replace functionality
- Zoom in/out support
- Font family and size customization
- Line and column position indicator
- Unsaved changes indicator
- Settings persistence

### Machine Learning Features
- Intelligent word suggestions based on:
  - User's typing history
  - Common word combinations
  - Context-aware predictions
  - Spell checking
  - Synonyms and related words
- Smart sentence suggestions using:
  - N-gram language models
  - TF-IDF vectorization
  - Cosine similarity matching
  - Learning from user's writing style
- Real-time learning and adaptation
- Persistent learning between sessions

## Requirements

- Python 3.x
- Required packages:
  - tkinter (usually comes with Python)
  - pyspellchecker
  - nltk
  - numpy
  - scikit-learn
  - pandas

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the text editor:
```
python text_editor.py
```

### Keyboard Shortcuts

- `Ctrl+N`: New file
- `Ctrl+O`: Open file
- `Ctrl+S`: Save file
- `Ctrl+Shift+S`: Save file as
- `Ctrl+F`: Find text
- `Ctrl+H`: Replace text
- `Ctrl++`: Zoom in
- `Ctrl+-`: Zoom out
- `Ctrl+0`: Reset zoom

### Features

1. **File Operations**
   - Create new files
   - Open existing files
   - Save files
   - Save files with a new name

2. **Text Formatting**
   - Change font family
   - Change font size
   - Apply bold formatting
   - Apply italic formatting
   - Apply underline formatting
   - Change text color

3. **Search and Replace**
   - Find text in the document
   - Replace text in the document

4. **Themes**
   - Nord theme (default)
   - Dark theme
   - Light theme

5. **View Options**
   - Zoom in/out
   - Reset zoom
   - Line and column position indicator

6. **Machine Learning Features**
   - **Word Suggestions**
     - Real-time word suggestions as you type
     - Based on your typing history and patterns
     - Spell checking and corrections
     - Synonym suggestions
     - Context-aware predictions
   
   - **Sentence Suggestions**
     - Smart sentence completions
     - Based on your writing style
     - Similar sentence suggestions
     - Learning from your previous sentences
   
   - **Learning System**
     - Continuously learns from your writing
     - Stores up to 1000 recent sentences
     - Updates models in real-time
     - Persists learning between sessions

## Settings and Data

The editor maintains several data files:

1. `editor_settings.json`
   - Font size
   - Font family
   - Selected theme

2. `word_frequency.json`
   - Word usage statistics
   - Personalized word suggestions

3. `ml_models.json`
   - N-gram language models
   - Sentence patterns
   - Writing style data

## How It Works

### Word Suggestions
1. As you type, the editor analyzes:
   - Current word context
   - Previous word combinations
   - Common patterns in your writing
2. Suggests words based on:
   - Spelling corrections
   - Your typing history
   - Contextual relevance
   - Synonym relationships

### Sentence Suggestions
1. The editor learns from your writing by:
   - Analyzing sentence structure
   - Identifying common patterns
   - Building a language model
2. Provides suggestions based on:
   - Similar sentences you've written
   - Common sentence patterns
   - Context and meaning
   - Writing style matching

### Learning Process
1. Real-time learning:
   - Captures word pairs (bigrams)
   - Analyzes word triplets (trigrams)
   - Stores complete sentences
2. Continuous improvement:
   - Updates models as you type
   - Adapts to your writing style
   - Improves suggestion accuracy
   - Maintains learning between sessions
