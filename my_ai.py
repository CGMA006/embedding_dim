import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB


# nltk.download('wordnet')
# nltk.download('punkt')

#Training sentences
sentences = [
    "Hello there!",
    "How are you?",
    "Goodbye!"
]

#Bot replies (used as labels)
labels = [
    "Hi! How can I help you?",
    "I’m doing well, thanks!",
    "Goodbye! Have a nice day!"
]

#lemmatizer
lemmatizer = WordNetLemmatizer()

# Step 3: Normalize function
def normalize(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    clean = [w for w in lemmas if w not in string.punctuation]
    return clean


normal = [normalize(s) for s in sentences]


all_words = []
for s in normal:
    all_words.extend(s)

vocab = sorted(set(all_words))
print("Vocabulary:", vocab)


def bag_of_words(sentence, vocab):
    sentence_words = normalize(sentence)
    vector = []
    for word in vocab:
        if word in sentence_words:
            vector.append(1)
        else:
            vector.append(0)
    return vector

# Convert all sentences to vectors
vectors = [bag_of_words(s, vocab) for s in sentences]
print("Vectors:", vectors)

#  Train model
model = MultinomialNB()
model.fit(vectors, labels)

# Step 7: Chatting with the bot
def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        vector = [bag_of_words(user_input, vocab)]
        prediction = model.predict(vector)
        print("Bot:", prediction[0])

# Start chat
chat()
