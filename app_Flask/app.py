from itertools import count
from flask import Flask, flash, redirect, render_template, request, session
from sympy import use
import tweepy
import os
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
from ensemble_model import combine_predictions

# Create an instance of Flask
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'reds209ndsldssdsljdsldsdsljdsldksdksdsdfsfsfsfit'

# Add Twitter API credentials
load_dotenv()
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')

# Authenticate with Twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
print("Authenticated with Twitter!")

# Route for the home page
@app.route('/')
def home():
    if not session.get('searched'):
        return render_template('search.html')
    else:
        labels = ["Positive", "Negative", "Neutral"]
        values = [session.get('positive', 0), session.get('negative', 0), session.get('neutral', 0)]
        colors = ["#8bc34a", "#ff5252", "#9e9e9e"]
        
        total = sum(values)
        if total > 0:
            percentages = [(value / total) * 100 for value in values]
        else:
            percentages = [0, 0, 0]
        
        usernames = session.get('usernames', [])
        counts = session.get('counts', [])
        total_tweets = session.get('total_tweets', 0)
        session['searched'] = False
        return render_template('chart.html', 
                               labels=labels, 
                               values=values, 
                               colors=colors, 
                               percentages=percentages, 
                               usernames=usernames, 
                               counts=counts,
                               total_tweets=total_tweets)

# Route for handling tweet search
@app.route('/search', methods=['POST'])
def do_search():
    search_query = request.form.get('search_query', '').strip()
    max_tweets = request.form.get('max_tweets', '').strip()

    if not search_query:
        flash('Search Query cannot be empty!')
        session['searched'] = False
    elif not max_tweets:
        flash('Max Tweets cannot be empty!')
        session['searched'] = False
    elif not max_tweets.isdigit():
        flash('Max Tweets should be a number!')
        session['searched'] = False
    else:
        max_tweets = int(max_tweets)
        if 0 < max_tweets <= 100000:
            tweets = []
            # try:
            #     for tweet in tweepy.Cursor(api.search_tweets, q=search_query, tweet_mode='extended', lang='en').items(max_tweets):
            #         tweets.append(tweet.full_text)
            # except tweepy.TweepyException as e:
            #     flash(f"Error: {str(e)}")
            #     return redirect('/')

            df = pd.read_csv('app_Flask/datasets/IndianElection19TwitterData.csv')
            df_filtered = df.sample(frac=1)
            df_filtered = df[df['Tweet'].str.contains(search_query)]
            df_filtered.drop(['Unnamed: 0'], axis=1, inplace=True)
            df_filtered.reset_index(drop=True, inplace=True)
            df_filtered.head()

            tweet_count = 0
            for tweet in df_filtered['Tweet']:
                tweets.append(tweet)
                tweet_count += 1
                if tweet_count >= max_tweets or tweet_count >= len(df_filtered):
                    break
            
            total_tweets = len(tweets)
            if not tweets:
                flash("No tweets found for the given query.")
                
            usernames = list(df_filtered['User'][:max_tweets])
            username_counts = Counter(usernames)
            top_usernames = username_counts.most_common(10)
            usernames, counts = zip(*top_usernames) if top_usernames else ([], [])

            # Initialize sentiment counts
            positive = 0
            negative = 0
            neutral = 0

            # Analyze sentiments
            for tweet in tweets:
                prediction = combine_predictions(tweet)
                if prediction > 0:
                    positive += 1
                elif prediction < 0:
                    negative += 1
                else:
                    neutral += 1

            # Store results in session
            session['positive'] = positive
            session['negative'] = negative
            session['neutral'] = neutral
            session['usernames'] = usernames
            session['counts'] = counts
            session['total_tweets'] = total_tweets
            session['searched'] = True
        else:
            flash('Max Tweets should be between 1 and 100000!')
            session['searched'] = False

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True, port=4000)
