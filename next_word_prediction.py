import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample corpus
corpus = [
    "I love machine learning",
    "I love deep learning",
    "I love natural language processing",
    "I enjoy coding in Python",
    "I enjoy solving problems"
]

# 1. Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus) #Assign unique number to each word
total_words = len(tokenizer.word_index) + 1
print(total_words)

# Most frequent gets smallest index number

# Rank 1 → "i"          (5 times) → Index 1
# Rank 2 → "love"       (3 times) → Index 2
# Rank 3 → "learning"   (3 times) → Index 3
# Rank 4 → "enjoy"      (2 times) → Index 4
# Rank 5 → "machine"    (1 time)  → Index 5
# Rank 6 → "deep"       (1 time)  → Index 6
# Rank 7 → "natural"    (1 time)  → Index 7
# Rank 8 → "language"   (1 time)  → Index 8
# Rank 9 → "processing" (1 time)  → Index 9
# Rank 10→ "coding"     (1 time)  → Index 10
# Rank 11→ "in"         (1 time)  → Index 11
# Rank 12→ "python"     (1 time)  → Index 12
# Rank 13→ "solving"    (1 time)  → Index 13
# Rank 14→ "problems"   (1 time)  → Index 14

# 2. Create input sequences and labels
input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)
print(input_sequences)

# 3. Pad sequences and split predictors/label
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_seq_len = max([len(x) for x in input_sequences])#to get maximum no.of words in a sentence
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
print(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
print(X)
# print(y)
y = to_categorical(y, num_classes=total_words)#one-hot encoding
print(y)

# 4. Build RNN model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
# model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train model
model.fit(X, y, epochs=200, verbose=1)

# 6. Function to predict next word
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    #max_seq_len-1 because we want to predict the next (final) word.
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    print(token_list)
    predicted = model.predict(token_list, verbose=0)
    print(predicted)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

# Example usage
seed = "I love machine"
print(f"Next word prediction: {predict_next_word(seed)}")

print(tokenizer.word_index)
