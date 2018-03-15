

```python
# dependencies
import tweepy
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt


# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


```


```python
#Twitter API Keys
from TwitterConfig import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)


#Setup Tweepy API Authentication

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#news outlets
NewsSource = ["@BBC","@CNN","@CBSNews","@nytimes","@FoxNews"]
```


```python
#pulling last 100 tweets from each organization and append to DataFrame

ID=[]
date=[]
text=[]
compound=[]
pos=[]
neu=[]
neg=[]
tweets_ago=[]

#loop through news sources
for Source in NewsSource:
    #loop through 5 pages of tweets (total 100 tweets)
    y=0
    for x in range(5):
        #get all tweets
        public_tweets=api.user_timeline(Source,page=x)
        print(f"Successfully pulled data for {Source}")
        #loop through each of the last 100 tweets
        for tweet in public_tweets:
            #append information into lists
            ID.append(Source)
            temp_date=tweet["created_at"]
            temp_date=datetime.strptime(temp_date,"%a %b %d %H:%M:%S %z %Y")
            date.append(temp_date)
            temp_text=tweet["text"]
            text.append(tweet["text"])
            results=analyzer.polarity_scores(temp_text)
            compound.append(results["compound"])
            pos.append(results["pos"])
            neu.append(results["neu"])
            neg.append(results["neg"])
            tweets_ago.append(y)
            y=y+1
            

            

            
                    
```

    Successfully pulled data for @BBC
    Successfully pulled data for @BBC
    Successfully pulled data for @BBC
    Successfully pulled data for @BBC
    Successfully pulled data for @BBC
    Successfully pulled data for @CNN
    Successfully pulled data for @CNN
    Successfully pulled data for @CNN
    Successfully pulled data for @CNN
    Successfully pulled data for @CNN
    Successfully pulled data for @CBSNews
    Successfully pulled data for @CBSNews
    Successfully pulled data for @CBSNews
    Successfully pulled data for @CBSNews
    Successfully pulled data for @CBSNews
    Successfully pulled data for @nytimes
    Successfully pulled data for @nytimes
    Successfully pulled data for @nytimes
    Successfully pulled data for @nytimes
    Successfully pulled data for @nytimes
    Successfully pulled data for @FoxNews
    Successfully pulled data for @FoxNews
    Successfully pulled data for @FoxNews
    Successfully pulled data for @FoxNews
    Successfully pulled data for @FoxNews
    


```python
#Create DataFrame
Sentiment_Data=pd.DataFrame({"News Source":ID,"Tweets Ago":tweets_ago, "Tweet":text,
                             "Tweet Date":date,"Compound List":compound,
                             "Positive List":pos,"Negative List":neg,
                            "Neutral List": neu})
```


```python
#create scatter plot
groups=Sentiment_Data.groupby("News Source")
fig, ax=plt.subplots()
ax.set_ylim(-1,1)
ax.set_xlim(110,-10)
ax.grid()
for name,group in groups:
    ax.plot(group["Tweets Ago"],group["Compound List"],alpha=0.5,marker="o",linestyle='',label=name)

ax.legend(loc='center left',bbox_to_anchor=(1,0.5),title="Media Source")    
today=datetime.today()
today=today.strftime("%Y-%m-%d")
plt.title(f"Sentiment Analysis of Tweets {today}")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.show()
```


![png](output_5_0.png)



```python
#group by News Source
News_Data=Sentiment_Data.groupby("News Source").mean()
News_Data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound List</th>
      <th>Negative List</th>
      <th>Neutral List</th>
      <th>Positive List</th>
      <th>Tweets Ago</th>
    </tr>
    <tr>
      <th>News Source</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.124133</td>
      <td>0.06097</td>
      <td>0.82998</td>
      <td>0.10909</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>@CBSNews</th>
      <td>-0.093463</td>
      <td>0.08127</td>
      <td>0.87408</td>
      <td>0.04463</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>0.076762</td>
      <td>0.06010</td>
      <td>0.85757</td>
      <td>0.08236</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.128850</td>
      <td>0.05817</td>
      <td>0.83971</td>
      <td>0.10211</td>
      <td>49.5</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>-0.049316</td>
      <td>0.07486</td>
      <td>0.86366</td>
      <td>0.06147</td>
      <td>49.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create bar chart
colors=["blue","orange","green","red","purple"]
fig, ax=plt.subplots()
rectangul=ax.bar(News_Data.index,News_Data["Compound List"],color=colors,alpha=0.5,align = "center")
ax.set_ylim(-.15,.15)
plt.title(f"Overall Media Sentiment based on Twitter ({today})")  
plt.ylabel("Tweet Polarity")
plt.show()
```


![png](output_7_0.png)


#Observable Trends

The data shows the following:
    1. on average media outlets try to be neutral in the wording of their tweets
    2. BBC and FoxNews tend to show the most positive news
    3. CBS and NYT tend to be more negative

