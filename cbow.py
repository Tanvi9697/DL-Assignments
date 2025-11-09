# -----------------------------
# 1. Import Libraries
# -----------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
import matplotlib.pyplot as plt

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
data = """The speed of transmission is an important point of difference between the two viruses. 
Influenza has a shorter median incubation period the time from infection to appearance of symptoms 
and a shorter serial interval the time between successive cases than COVID 19 virus. 
The serial interval for COVID 19 virus is estimated to be 5 6 days while for influenza virus  
the serial interval is 3 days. This means that influenza can spread faster than COVID-19."""

sentences = data.lower().split('.')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# create word-index mappings for prediction helper
word_to_index = tokenizer.word_index
index_to_word = {v: k for k, v in word_to_index.items()}

# -----------------------------
# 3. Create Training Data (Context → Target)
# -----------------------------
context_size = 2
contexts, targets = [], []

for seq in sequences:
    for i in range(context_size, len(seq) - context_size):
        target = seq[i]
        context = [seq[i - 2], seq[i - 1], seq[i + 1], seq[i + 2]]
        contexts.append(context)
        targets.append(target)

X = np.array(contexts)
Y = np.array(targets)

# -----------------------------
# 4. Build Model
# -----------------------------
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=20, input_length=4),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(X, Y, epochs=100, verbose=0)

# -----------------------------
# 6. Plot Accuracy & Loss
# -----------------------------
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title("CBOW Model Training")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()

# -----------------------------
# 7. Display Word Embeddings
# -----------------------------
embeddings = model.layers[0].get_weights()[0]
print("\nWord Embeddings:\n")
print(embeddings)

# -----------------------------
# 8. Simple Prediction Helper
# -----------------------------
def guess(word):
    if word in word_to_index:
        x = np.array([[word_to_index[word]] * 4])
        p = model.predict(x, verbose=0)
        print(word, "→", index_to_word[np.argmax(p)])
    else:
        print("word not found")

# Example usage
guess("virus")