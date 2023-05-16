from flask import Flask, jsonify, request
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load the pickled model
with open('svm_pickle_model.pkl', 'rb') as f:
    svm_dict = pickle.load(f)

# Extract the SVM model and feature names from the dictionary
svm_model = svm_dict["model"]
feature_names = svm_dict["feature_names"]

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

@app.route('/predict', methods=['POST'])
def predict():
    input_array = request.get_json()['input_array']

    # Perform sentiment analysis and store the results in a list
    sentiment_scores = []
    for text in input_array:
        score = sid.polarity_scores(str(text))
        if(score['compound'] > 0):
            sentiment_scores.append(1)
        elif(score['compound'] == 0):
            sentiment_scores.append(0)
        else:
            sentiment_scores.append(-1)

    # Predict the disorder type using the SVM model
    predictions = svm_model.predict([sentiment_scores])
    disorder = predictions[0]

    # Return the disorder type as a JSON response
    return jsonify({'disorder': disorder})

if __name__ == '__main__':
    app.run(debug=True)
