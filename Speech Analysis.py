import pandas as pd
import numpy as np
dataset=pd.read_csv('path/to/datset')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
corpus=[]
for i in range(0,50000):
        review=re.sub('[^a-zA-Z]',' ',dataset['review'][i])
        review= review.lower()
        review= review.split()
        ps=PorterStemmer()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        corpus.append(review)
        print(i)

#Creating Bag of Word
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=500)
x=cv.fit_transform(corpus).toarray()
#import label dataset
y=dataset.iloc[:,1].values

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
classifier.fit(x,y)
#save model
import pickle
pickle.dump(classifier,open('classifiers.sav','wb'))

text=[]
import speech_recognition as sr
filename='path/to/audio file'
r = sr.Recognizer()

with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        print(text)
      
text=re.sub('[^a-zA-Z]',' ',text)
text= text.lower()
text= text.split()
ps=PorterStemmer()
text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text=' '.join(text)
 
text=text.split('delimiter')


test=cv.transform(text).toarray()

y_pred=classifier.predict(test[0,:].reshape(1,-1))
   
print(y_pred) 