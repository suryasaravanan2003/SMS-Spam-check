# SMS-Spam-check
SMS Spam Detection using Python and Machine Learning


🧩 Description:

In this project, we develop an SMS Spam Classifier that identifies unwanted spam messages from normal texts.
We use a dataset of labeled SMS messages (spam and ham) and train a Naïve Bayes Classifier to recognize spam based on message content.

The model is built using text preprocessing, tokenization, stop word removal, and vectorization (Bag of Words) to convert messages into numerical form that the machine learning algorithm can process.

The user can input any message, and the system will predict whether it is Spam or Not Spam.

⚙️ Libraries Used:
Library	Purpose
pandas	For loading and handling the dataset (CSV file)
nltk	For natural language processing tasks like removing stopwords
scikit-learn (sklearn)	For vectorizing text data (CountVectorizer), splitting data, and training Naive Bayes model
matplotlib / seaborn (optional)	For visualization or accuracy graph
numpy	For numerical operations (used internally by sklearn)

🔬 Modules Used:

Data Collection:
Load dataset (spam.csv) containing SMS messages labeled as “spam” or “ham”.

Data Preprocessing:

Remove missing values

Convert text to lowercase

Remove stopwords (using NLTK)

Feature Extraction:
Convert text messages into numerical features using CountVectorizer (Bag of Words model).

Model Building:
Use Multinomial Naive Bayes, a probabilistic model suitable for text classification.

Model Evaluation:
Measure model accuracy on test data.

User Interface:
Allow user input (SMS message) and display prediction result → “SPAM” or “NOT SPAM”.
