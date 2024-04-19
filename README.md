# Sentiment Analysis on Amazon Reviews

### Overview
This project focuses on sentiment analysis of product reviews from Amazon, particularly targeting the JBL Flip Black Noise Cancelling Speaker. By employing machine learning techniques, we aim to classify reviews into positive, negative, or neutral sentiments. This analysis can provide valuable insights for product evaluation and customer feedback assessment.

### Introduction
In today's digital age, online reviews play a crucial role in influencing consumer decisions. Understanding the sentiment expressed in these reviews can help businesses gauge customer satisfaction and make informed decisions regarding product improvements or marketing strategies. This project leverages natural language processing (NLP) and machine learning to analyze sentiment in Amazon product reviews.

### Requirements
To run this project, you need the following libraries installed:
* numpy
* pandas
* requests
* beautifulsoup4
* nltk
* scikit-learn
* matplotlib
* seaborn

### Data Collection
We utilize web scraping techniques with BeautifulSoup and requests libraries to extract product reviews from Amazon. Specifically, we target reviews for the JBL Flip Black Noise Cancelling Speaker from multiple pages of Amazon's review section.

### Data Preprocessing
* The extracted reviews are cleaned and organized into a structured format using Pandas DataFrame.
* Ratings are categorized as 'positive', 'negative', or 'neutral' based on predefined thresholds.
* Null values, if any, are removed to ensure data integrity.

### Model Training
* The reviews undergo tokenization and stemming using NLTK (Natural Language Toolkit) to prepare them for analysis.
* The dataset is split into training and testing sets with an 80/20 ratio for model evaluation.
* Text data is transformed into numerical vectors using CountVectorizer.
* We train multiple classifiers, including:
    * Naive Bayes classifier (MultinomialNB)
    * Support Vector Machine (SVM) classifier
    * Random Forest classifier
* Each classifier is trained on the vectorized data to predict sentiment labels.

### Evaluation
* Performance metrics such as accuracy, precision, recall, and F1-score are computed to assess the effectiveness of the trained classifier.
* Confusion matrix and visualizations are generated to provide a comprehensive view of the model's performance.

### Conclusion
This project showcases the application of machine learning techniques for sentiment analysis of Amazon product reviews. Businesses can gain valuable insights into customer opinions and sentiments by accurately classifying reviews into positive, negative, or neutral sentiments, facilitating data-driven decision-making processes.

