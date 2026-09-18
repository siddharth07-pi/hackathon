import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import tokenize
import io
import sys
import json # Added import for json module

# --- 1. Data Loading ---
print('--- 1. Data Loading ---')
file_path = r'C:\Users\DELL\.gemini\antigravity\scratch\dataset_generator\python_bug_pairs_10k.jsonl' # Update this path if your file is in a different location

# Read the file content as a string first, then use io.StringIO for robustness
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        json_content = f.read()

    if not json_content.strip(): # Check if content is empty or just whitespace
        raise ValueError(f"File '{file_path}' is empty or contains only whitespace.")

    df_dataset = pd.read_json(io.StringIO(json_content), lines=True)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's uploaded to /content/.")
    # Initialize an empty DataFrame to prevent subsequent errors if file is missing
    df_dataset = pd.DataFrame({'faulty_code': [], 'error_category': [], 'is_faulty': []})
    # Re-raise the error to halt execution for user attention
    raise
except ValueError as ve:
    print(f"Error reading JSONL file (likely malformed JSON or content issue): {ve}")
    df_dataset = pd.DataFrame({'faulty_code': [], 'error_category': [], 'is_faulty': []})
    raise
except Exception as e:
    print(f"An unexpected error occurred during file loading: {e}")
    df_dataset = pd.DataFrame({'faulty_code': [], 'error_category': [], 'is_faulty': []})
    raise

df_dataset['is_faulty'] = True # Assuming all entries are faulty as per original context

print(f"Dataset loaded from {file_path}. Total samples: {len(df_dataset)}")
print(f"Unique error categories: {df_dataset['error_category'].unique()}\n")

# --- 2. Feature Extraction ---
print('--- 2. Feature Extraction ---')
def python_tokenizer(code_string):
    """
    Tokenizes a Python code string using the 'tokenize' module.
    Removes comments and whitespace tokens.
    """
    tokens = []
    try:
        # The tokenize module expects a readline-like method
        token_stream = tokenize.generate_tokens(io.StringIO(code_string).readline)
        for toknum, tokval, _, _, _ in token_stream:
            # Ignore comments and whitespace
            if toknum == tokenize.COMMENT or toknum == tokenize.NEWLINE or toknum == tokenize.NL or toknum == tokenize.INDENT or toknum == tokenize.DEDENT or toknum == tokenize.ENCODING or toknum == tokenize.ENDMARKER:
                continue
            # Convert identifier tokens to a generic placeholder for simplicity, or keep them
            tokens.append(tokval)
    except tokenize.TokenError:
        # Handle cases where code might be malformed and tokenize fails
        return []
    return tokens

df_dataset['tokenized_code'] = df_dataset['faulty_code'].apply(lambda x: ' '.join(python_tokenizer(x)))
print("Code tokenization complete.")

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x.lower(), token_pattern=None)
X_features = vectorizer.fit_transform(df_dataset['tokenized_code'])
y_error_type = df_dataset['error_category']

print(f"Feature matrix shape: {X_features.shape}")
print(f"Number of unique tokens (vocabulary size): {len(vectorizer.vocabulary_)}")
print("Feature extraction (Bag-of-Words) complete.\n")

# --- 3. Model Training ---
print('--- 3. Model Training ---')
X_train, X_test, y_train, y_test = train_test_split(X_features, y_error_type, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)
print("Model training complete.\n")

# --- 4. Model Evaluation ---
print('--- 4. Model Evaluation ---')
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Helper function to execute code and capture errors
def execute_code_and_capture_error(code_str):
    """
    Executes a Python code string and captures any exceptions.
    Returns the error type and error message if an error occurs, otherwise None.
    """
    original_stderr = sys.stderr
    redirected_stderr = io.StringIO()
    sys.stderr = redirected_stderr

    try:
        exec(code_str)
        return None, None
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        return error_type, error_message
    finally:
        sys.stderr = original_stderr # Restore original stderr

# Updated function to predict error type and also provide actual error details
print('\n--- Optional: Prediction Function (with Error Details) ---')
def predict_error_details(code_snippet):
    # 1. Predict error category using the ML model
    tokenized_snippet = ' '.join(python_tokenizer(code_snippet))
    # Use the same vectorizer to transform new code
    snippet_features = vectorizer.transform([tokenized_snippet])
    predicted_category = classifier.predict(snippet_features)[0]

    # 2. Execute code to capture actual error details
    actual_error_type, actual_error_message = execute_code_and_capture_error(code_snippet)

    #return predicted_category, actual_error_type, actual_error_message
    print(f"Predicted Category: {predicted_category}")
    print(f"Actual Error Type: {actual_error_type}")
    print(f"Actual Error Message: {actual_error_message}")

Code = input("enter a python code snippet to predict its error type and correction message:")
predict_error_details(code_snippet = Code)


