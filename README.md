# sentiment-analysis-twitter

This project is an interactive web application developed using Streamlit, designed to perform sentiment analysis on text data. It utilizes a pre-trained Naive Bayes classifier to determine the sentiment of a given input text. The application enables users to input text and receive the predicted sentiment, which can be classified as negative, neutral, or positive. Through this application, users can gain insights into the sentiment conveyed by their text inputs.

# Dataset
The sentiment analysis model is trained on a dataset of tweets. The dataset, stored in a file named tweets.csv, contains text data along with corresponding sentiment labels. The dataset is preprocessed to filter out entries with low confidence scores, and the text data is cleaned by removing stopwords, punctuation, and converting words to their base forms using lemmatization.

Find the dataset used at - [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

# Model Training and Evaluation
The cleaned text data is transformed into numerical features using the CountVectorizer from scikit-learn. The vectorized data is then split into training and testing sets. A Multinomial Naive Bayes classifier is trained on the training set and evaluated on the testing set. The performance of the model is evaluated using a classification report, which includes precision, recall, F1-score, and support for each sentiment class.

# Deployment
The trained model is saved as a pickle file named sent_model.pkl. The Streamlit application loads this saved model for sentiment prediction. The application allows users to input text and obtain the predicted sentiment by leveraging the trained model.

# Usage
To run the sentiment analysis app, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/pwnbisht/sentiment_analysis.git
   
2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    
3. Run the Streamlit app:

   ```shell
   streamlit run app.py
   
**Note:** Access the app in your browser at http://localhost:8501.

# Requirements
- Python 3.7 or above
- pandas
- nltk
- scikit-learn
- streamlit
