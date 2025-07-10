from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import re
import joblib

# Load and prepare data
data = pd.read_csv("./data/filtered_dataset.csv")

def extract_features_from_appname(appname):
    """Extract meaningful features from app package name"""
    # Split by dots and get individual components
    parts = appname.lower().split('.')
    
    # Extract domain-like features
    domain_part = parts[0] if parts else ''
    app_part = parts[-1] if len(parts) > 1 else ''
    middle_parts = '.'.join(parts[1:-1]) if len(parts) > 2 else ''
    
    # Game-related keywords
    game_keywords = [
        'game', 'play', 'battle', 'fight', 'war', 'quest', 'adventure', 'rpg',
        'puzzle', 'match', 'candy', 'saga', 'clash', 'craft', 'build', 'racing',
        'shooter', 'action', 'strategy', 'simulation', 'casino', 'poker', 'slots',
        'zombie', 'dragon', 'ninja', 'hero', 'legend', 'fantasy', 'magic', 'royal',
        'king', 'empire', 'tower', 'defense', 'attack', 'combat', 'arena', 'league',
        'championship', 'tournament', 'sport', 'football', 'soccer', 'basketball',
        'tennis', 'golf', 'racing', 'car', 'bike', 'truck', 'plane', 'ship',
        'space', 'galaxy', 'star', 'planet', 'alien', 'robot', 'mech', 'tank',
        'gun', 'weapon', 'sword', 'bow', 'arrow', 'bomb', 'missile', 'laser',
        'treasure', 'gold', 'diamond', 'jewel', 'coin', 'reward', 'prize',
        'level', 'stage', 'boss', 'enemy', 'monster', 'creature', 'beast',
        'pixel', 'retro', 'arcade', 'classic', 'mini', 'casual', 'hyper',
        'super', 'mega', 'ultra', 'extreme', 'pro', 'master', 'champion',
        'world', 'universe', 'realm', 'land', 'island', 'city', 'town',
        'farm', 'garden', 'house', 'home', 'family', 'pet', 'animal',
        'fish', 'bird', 'cat', 'dog', 'horse', 'cow', 'pig', 'sheep',
        'run', 'jump', 'fly', 'swim', 'drive', 'ride', 'walk', 'move',
        'speed', 'fast', 'quick', 'rush', 'dash', 'boost', 'turbo',
        'fun', 'enjoy', 'entertainment', 'leisure', 'hobby', 'activity'
    ]
    
    # Check for game keywords in the full app name
    full_name = appname.lower()
    game_keyword_count = sum(1 for keyword in game_keywords if keyword in full_name)
    
    # Additional features
    features = {
        'full_name': appname.lower(),
        'domain_part': domain_part,
        'app_part': app_part,
        'middle_parts': middle_parts,
        'num_parts': len(parts),
        'game_keyword_count': game_keyword_count,
        'has_game_keywords': game_keyword_count > 0,
        'length': len(appname),
        'has_numbers': bool(re.search(r'\d', appname)),
        'has_special_chars': bool(re.search(r'[^a-zA-Z0-9.]', appname))
    }
    
    return features

# Extract features and labels
print("Extracting features from app names...")
X = data['appname']
y = data['categorygame'].apply(lambda x: 1 if x.strip().lower() == 'game' else 0)

print(f"Dataset size: {len(X)}")
print(f"Games: {sum(y)}, Non-games: {len(y) - sum(y)}")
print(f"Game ratio: {sum(y)/len(y):.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create multiple models for comparison
models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# Train and evaluate models
best_model = None
best_score = 0
best_name = ""

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Keep track of best model
    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = model
        best_name = name
    
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))

print(f"\n🏆 Best model: {best_name} (Accuracy: {best_score:.4f})")

# Hyperparameter tuning for the best model type
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

if best_name == 'Logistic Regression':
    param_grid = {
        'tfidf__ngram_range': [(1, 2), (1, 3), (2, 3)],
        'tfidf__max_features': [5000, 10000, 15000],
        'clf__C': [0.1, 1, 10]
    }
    base_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
else:
    param_grid = {
        'tfidf__ngram_range': [(1, 2), (1, 3), (2, 3)],
        'tfidf__max_features': [5000, 10000, 15000],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [10, 20, None]
    }
    base_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

print("Performing grid search...")
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Final model evaluation
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"\nFinal model test accuracy: {final_accuracy:.4f}")
print("\nFinal Classification Report:")
print(classification_report(y_test, final_predictions))

# Save the best model
joblib.dump(final_model, 'app_game_classifier_model.pkl')
print("\n✅ Model saved as 'app_game_classifier_model.pkl'")

# Test on example data
print("\n" + "="*50)
print("TESTING ON EXAMPLE DATA")
print("="*50)

examples = [
    "com.infinitygames.hex",                 # likely game
    "com.quicknotes.notepad",               # not game
    "com.speedracer.extreme3d",             # game
    "com.taskbuddy.scheduler",              # not game
    "com.dragonking.battlearena",           # game
    "com.weatherwise.forecastpro",          # not game
    "com.minijump.frenzy",                  # game
    "com.dailytodo.listmaker",              # not game
    "com.warofclans.empiredefense",         # game
    "com.meditate.relaxnow",                # not game
    "com.crashzombies.apocalypse",          # game
    "com.calculatorplus.tools",             # not game
    "com.sandboxcraft.buildnexplore",       # game
    "com.passwordmanager.lockvault",        # not game
    "com.piratelegends.caribbeantreasure",  # game
    "com.ebookreader.libpremium",           # not game
    "com.fps.shootingmissionelite",         # game
    "com.videoplayer.ultra",                # not game
    "com.wizardduel.magicwars",             # game
    "com.scanit.documentscanner",           # not game
]

predictions = final_model.predict(examples)
prediction_probabilities = final_model.predict_proba(examples)

print("Predictions with confidence scores:")
print("-" * 70)
for pkg, label, prob in zip(examples, predictions, prediction_probabilities):
    confidence = max(prob)
    result = 'Game' if label == 1 else 'Not Game'
    print(f"{pkg:<45} -> {result:<8} (confidence: {confidence:.3f})")

# Feature importance analysis (if using Random Forest)
if hasattr(final_model.named_steps['clf'], 'feature_importances_'):
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Get feature names from TF-IDF vectorizer
    feature_names = final_model.named_steps['tfidf'].get_feature_names_out()
    importances = final_model.named_steps['clf'].feature_importances_
    
    # Sort features by importance
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 20 most important features:")
    for i, (feature, importance) in enumerate(feature_importance_pairs[:20]):
        print(f"{i+1:2d}. {feature:<20} : {importance:.4f}")

# Function to classify new apps
def classify_app(model, app_name):
    """Classify a single app as game or not game"""
    prediction = model.predict([app_name])[0]
    probability = model.predict_proba([app_name])[0]
    confidence = max(probability)
    
    result = 'Game' if prediction == 1 else 'Not Game'
    return result, confidence

print("\n" + "="*50)
print("CLASSIFICATION FUNCTION EXAMPLE")
print("="*50)

# Example usage
test_apps = [
    "com.supercell.clashofclans",
    "com.whatsapp",
    "com.king.candycrushsaga",
    "com.google.android.apps.photos"
]

for app in test_apps:
    result, confidence = classify_app(final_model, app)
    print(f"{app} -> {result} (confidence: {confidence:.3f})")

print("\n🎯 Classification complete! Use the saved model for future predictions.")
