import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Rename columns
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
data = data[['label', 'message']]

# Drop missing or empty rows
data.dropna(subset=['message', 'label'], inplace=True)

# Convert to string and lowercase
data['message'] = data['message'].astype(str)
data['message'] = data['message'].str.lower()

# Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# --- USER INPUT ---
print("\n=== SMS SPAM DETECTOR ===")
user_input = input("Enter an SMS message: ")

user_input_transformed = cv.transform([user_input])
prediction = model.predict(user_input_transformed)[0]

if prediction == 'spam':
    print("\n🚨 This message is SPAM.")
else:
    print("\n✅ This message is NOT SPAM.")
