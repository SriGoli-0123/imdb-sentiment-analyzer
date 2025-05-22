# IMDB Sentiment Analysis

A sentiment analysis project using Hugging Face Transformers pipeline for IMDB movie reviews. For my personal practice purpose only.

## Requirements
- Python 3.8+
- Required packages: See requirements.txt

## Installation
```bash
git clone https://github.com/SriGoli-0123/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install --no-cache-dir -r requirements.txt
```

## Usage

1. **Evaluate Pre-trained Model**:
```bash
python evaluate.py
```

2. **Run Streamlit App**:
```bash
streamlit run app.py
```

## Project Structure
- `evaluate.py`: Evaluates model performance on IMDB test set
- `app.py`: Streamlit web interface for predictions
- `requirements.txt`: Python dependencies
