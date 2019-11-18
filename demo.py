import SentimentClassifier
import nltk
from nltk import sent_tokenize

vad_classifier = SentimentClassifier.VADClassifier()
while True:
    text = raw_input("Enter text to be classified: ")
    for sentence in sent_tokenize(text):
        vad = vad_classifier.analyzeSentiment(sentence)
        print(sentence)
        print('valence, arousal and dominance: ', vad)

