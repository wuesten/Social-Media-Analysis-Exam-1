<h1><center>«Portfolio-Exam Part I» </center></h1>
<h2><center>MADS-SMA </center></h2>
<h3><center>Author: Tom Wüsten </center></h3>

### Introduction
This paper deals about the evaluation from different approaches of sentiment analysis. The basis for this is a dataset of restaurant reviews from the yelp platform. The reviews are in English and rate restaurants in Hamburg. The rating is done by stars from 1-5, where 5 stars is the best rating. At the beginning, information about the data set will be gathered in the data exploration, which will be used for the pre-processing of the data. Based on this a Sentiment analysis will compare classic approaches like Bag of Words with newer approaches like Transformer. Furthermore, it will be discussed which methods can be used to train Transformer from the platform Huggingface on own data.

### Data Exploration
In the data analysis we want to get a picture of how the data is distributed. For this purpose, we want to look at the structure of the ratings and give an overview of the period in which the comments were created. <br>
Based on the first graph, we can see that the ratings of the restaurants are often good. The most votes have received 5 & 4 stars. It is remarkable that there are few bad ratings. In the second plot, we examine the text length of the comments. Most of the comments are in the range between 100 words. <br>
<img src="/output/distribution_text_rating.png" alt="Alt text" title="Optional title">

The plot below examines when comments have been written. The dataset includes ratings from 2006 to 2022, with the majority of ratings from 2015 & 2016. Furthermore, it can be seen that the months do not have a large influence on the visits. There is only a drop in ratings in November and December.<br>

<img src="/output/distribution_text_rating.png" alt="Alt text" title="Optional title">

### Results

<img src="/output/conf_overview.png" alt="Alt text" title="Optional title">
