#!/usr/bin/env python3
# coding: utf-8

# ### Import Packages

# In[2]:
from caci_functions import *


tweet_csv = 'test_tweets_7.csv'
user_csv = 'users_table3.csv'


# In[16]:


test_tweets = pd.read_csv(tweet_csv)


# In[17]:

today = date.today()
yesterday = today - timedelta(days = 2)
today = str(today)
yesterday = str(yesterday)
mask = (test_tweets['date'] >= yesterday) & (test_tweets['date'] <= today)

# In[18]:



mask_tweets = test_tweets.loc[mask]
mask_tweets

time_now = datetime.now()
final_timestamp = time_now.strftime("%b-%d-%Y")

final = trending_topics(mask_tweets,50)
final.to_csv(f'test_topics_{final_timestamp}.csv', index=False)
print('it is done my man')


# In[ ]:




