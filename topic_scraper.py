#!/usr/bin/env python3
# coding: utf-8

# ### Import Packages

from caci_functions import *


date = datetime.now()
date_csv =date.strftime("%b-%d-%Y")


#cities = cities_dict()
topic = 'trending'
result_type = 'recent'
radius: str = '15km'
count = 1200

top_topics = ['black history','euphoria','jeen-yuhs']

for t in top_topics:
    tweet_topic_scrape(t,topic,result_type,count)
    print(f'done with {t} topic')