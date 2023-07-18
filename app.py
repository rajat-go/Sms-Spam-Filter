from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__, template_folder='templates')

# Load the SMS spam collection dataset
data = pd.read_csv('spam.csv', encoding='latin1')
X = data['message']
y = data['label']

# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect-spam', methods=['POST'])
def detect_spam():
    message = request.form['message']
    sms_features = vectorizer.transform([message])
    prediction = clf.predict(sms_features)[0]
    if prediction == 'spam':
        result_message = "This message is identified as SPAM."
    else:
        result_message = "This message is NOT spam."

    return render_template('result.html', result_message=result_message)

if __name__ == '__main__':
    app.run()
