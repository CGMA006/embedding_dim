import nltk
import string
from nltk.stem import WordNetLemmatizer
import numpy as np

# nltk.download('wordnet')
# nltk.download('punkt')

sentences = ["Hello there!", "How are you?", "Goodbye!"]

#lemmatizer
lemmatizer = WordNetLemmatizer()

#Normalize function
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
word_to_idx = {w: i for i, w in enumerate(vocab)}
print("Vocabulary:", vocab)


embedding_dim = 5
np.random.seed(0)
W_embed = np.random.randn(len(vocab), embedding_dim)

def sentence_to_vector(sentence):
    tokens = normalize(sentence)
    vecs = [W_embed[word_to_idx[w]] for w in tokens if w in word_to_idx]
    if vecs:
        return np.mean(vecs, axis=0)  # simple average of embeddings
    else:
        return np.zeros(embedding_dim)

# Test
for s in sentences:
    vec = sentence_to_vector(s)
    print(f"Sentence: {s} → Vector: {vec}")


