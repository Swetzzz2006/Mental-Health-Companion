from textblob import TextBlob

text = "I feel very sad today."
analysis = TextBlob(text)

if analysis.sentiment.polarity > 0:
    print("Positive")
elif analysis.sentiment.polarity < 0:
    print("Negative")
else:
    print("Neutral")

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

result = classifier("I feel stressed and overwhelmed.")
print(result)
