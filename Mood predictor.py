import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    'Text': [
        'I am very happy today',
        'This is the best day ever',
        'I feel sad and lonely',
        'I am disappointed',
        'I am angry with my friend',
        'This makes me furious',
        'I am worried about exams',
        'I feel nervous',
        'Everything is normal',
        'It is an ordinary day'
    ],
    'Mood': [
        'Happy', 'Happy',
        'Sad', 'Sad',
        'Angry', 'Angry',
        'Anxious', 'Anxious',
        'Neutral', 'Neutral'
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Mood']

model = MultinomialNB()
model.fit(X, y)

text = input("Enter your message: ")

text_vector = vectorizer.transform([text])
prediction = model.predict(text_vector)

print("Predicted Mood:", prediction[0])
