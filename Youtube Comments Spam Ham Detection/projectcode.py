# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 06:13:52 2022

@author: 16472,niyanta,chishtpher
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#loading the data set 



data = pd.read_csv("C:/Users/16472/Desktop/Comp 237/YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv", )




     

#display the data 


data.head()


# Display the column names
print(data.columns)



#Data Exploratoion
# checking if the data set is balanced or unbalanced
print("Count of Classes")
count =(data.groupby("CLASS")["CLASS"].count())
print(count)



# finding if we have same account spaming multiple times
print("TOP 10 AUTHOR who have commented more then once: ")
print(data.groupby("AUTHOR")["CLASS"].count().sort_values(ascending=False).head(10))

#count nuber of unique
data["AUTHOR"].nunique()

# import the nlp Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



#getting english stop words 
stop_words =set(stopwords.words('english'))



# converting the data set to lower case

data['CONTENT'] =data['CONTENT'].str.lower()




# removing stop words
data['CONTENT'] = data['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))



def stars(row):
    if row.CONTENT :

        row.CONTENT = row.CONTENT.replace(".", " ")
        row.CONTENT = row.CONTENT.replace("/", " ")
        row.CONTENT = row.CONTENT.replace("  ", " ")
        row.CONTENT = row.CONTENT.replace(";", " ")
        row.CONTENT = row.CONTENT.replace("%", " ")
        row.CONTENT = row.CONTENT.replace("!", " ")
        row.CONTENT = row.CONTENT.replace(":", " ")
        row.CONTENT = row.CONTENT.replace("?", " ")
       
        
        return row.CONTENT
    

data['CONTENT'] = data.apply(stars, axis='columns')


#shuffle the dataset

data.sample(frac=1, replace=True, random_state=70)


from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer





count_vect = CountVectorizer()
final_counts = count_vect.fit_transform(data['CONTENT'].values)



final_counts.get_shape()



final_counts=final_counts.toarray()

type(final_counts)


tfidf = TfidfTransformer()
X_data_full_tfidf = tfidf.fit_transform(final_counts).toarray()



X =X_data_full_tfidf
Y = data['CLASS']

  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state= 70)


MNB = MultinomialNB()
MNB.fit(X_train, y_train)
predictions = MNB.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
round((accuracy_score(y_test, predictions)*100),2)


#corss validation score 
from sklearn.model_selection import cross_val_score
print(cross_val_score(MNB,X,Y))
print("Mean is : ")
print(cross_val_score(MNB,X,Y).mean())


print("Accuracy is:" )
print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))





# Come up with 6 comments....
to_pred = ["please subscribe to me is a spam ",
                    
          "If she really did this there",
           "This is the absolute best she has ever looked. What an incredible song extremely catchy with unique lyrics This mesmerizing video is an instant classic",
           "Shes back I've been waiting for these vibes for so looooooongggggggg Thank you goddess",
           "If she had sung a few more seconds Simon would've started crying tears of joy",
           "check out youtube"
           "LMFAO IS THE BEST"
           ]

final_counts1 = count_vect.transform(to_pred)


final_counts1=final_counts1.toarray()

input6 = tfidf.transform(final_counts1).toarray()
predictions1 = MNB.predict(input6)
print("Output:")
    
print(predictions1)
x = zip(to_pred,final_counts1)
print(x)

# plot the graphs 


from wordcloud import WordCloud
bot = data["CLASS"]==1
data_bot = data[bot]



text = " ".join(cat.split()[0] for cat in data['CONTENT'])

data_human = data[data["CLASS"]==0]

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.figure(figsize=(15, 15))
plt.imshow(word_cloud, interpolation='bilinear')


plt.axis("off")
plt.show()


# plot the graphs Bots


plt.figure(figsize=(15, 15))
text = " ".join(cat.split()[0] for cat in data_bot['CONTENT'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# plot the graphs  humans

text = " ".join(cat.split()[0] for cat in data_human['CONTENT'])

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
plt.figure(figsize=(15, 15))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()





    





