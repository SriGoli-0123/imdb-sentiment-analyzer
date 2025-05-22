from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load dataset
dataset = load_dataset('imdb')
test_data = dataset['test'].shuffle().select(range(1000))  # Use subset for faster evaluation

# Load pre-trained pipeline
classifier = pipeline('sentiment-analysis')

# Get predictions and true labels
true_labels = test_data['label']
predictions = []
for text in test_data['text']:
    result = classifier(text[:512])[0]  # Truncate to 512 tokens for model
    predictions.append(1 if result['label'] == 'POSITIVE' else 0)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
