import csv
from textblob import TextBlob

# tweets taken directly from twitter for the purpose of this program
with open('tweets.txt', 'r') as tweets_file:
    tweets = tweets_file.readlines()

with open('LPfDS_2_analysis.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['tweet', 'polarity', 'subjectivity'])
    for tweet in tweets:
        analysis = TextBlob(tweet.rstrip()).sentiment
        csv_writer.writerow([tweet.rstrip(), analysis.polarity, analysis.subjectivity])
