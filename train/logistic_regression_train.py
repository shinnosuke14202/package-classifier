from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load and prepare data
data = pd.read_csv("./data/filtered_dataset.csv")

# Extract features and labels
X = data['appname']
y = data['categorygame'].apply(
    lambda x: 1 if x.strip().lower() == 'game' else 0)

print(f"Dataset size: {len(X)}")
print(f"Games: {sum(y)}, Non-games: {len(y) - sum(y)}")
print(f"Game ratio: {sum(y)/len(y):.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Create model: takes all word -> add them to list and weight them (assign a number)
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # Use 1-word, 2-word, and 3-word combinations
        max_features=10000,  # use only 10000 most appearance words
        stop_words='english',  # remove stop words, word that not really affect sentences
        lowercase=True,
        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
    )),
    # random state = 42 for reproducible weight initialization / shuffling, iter:  is the optimization iteration limit
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# Cross-validation: Splits training data into 5 folds, trains on 4, tests on 1 (repeated 5 times)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(
    f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train on full training set
model.fit(X_train, y_train)

# Test set evaluation
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Save the model
joblib.dump(model, './model/linear_regression_model_v0.pkl')
print("✅ Model saved as 'linear_regression_model_v0.pkl'")

param_grid = {
    # ngram_range: Whether to use single words, pairs, or triplets
    'tfidf__ngram_range': [(1, 2), (1, 3), (2, 3)],
    # max_features: How many most frequent words to keep
    'tfidf__max_features': [5000, 10000, 15000],
    'clf__C': [0.1, 1, 10]  # For Logistic Regression
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Test set evaluation using the tuned model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy after tuning: {test_accuracy:.4f}")

# Save the model
joblib.dump(best_model, './model/linear_regression_model_v1.pkl')
print("✅ Model saved as 'linear_regression_model_v1.pkl'")
