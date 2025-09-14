# 📖 Text Generation using LSTM

This project demonstrates how to build a **text generation model** using a **Bidirectional LSTM** in Keras/TensorFlow. The model is trained on a classic text from **Project Gutenberg** and can generate new text in a similar style.

---

## 🔹 Dataset

We use **Sherlock Holmes** text from Project Gutenberg:

- Download URL: [https://www.gutenberg.org/files/1661/1661-0.txt](https://www.gutenberg.org/files/1661/1661-0.txt)  
- Saved locally as `book.txt`.

The text is preprocessed by:

1. Converting all text to lowercase.
2. Splitting the text into sentences.

---

## 🔹 Text Preprocessing

1. **Tokenization** – Convert words into integer indices using Keras `Tokenizer`.
2. **Sequence Creation** – Create input sequences of words (n-grams) for training.
3. **Padding** – Pad sequences to the same length using `pad_sequences`.
4. **Labels** – Last word of each sequence is treated as the target label.
5. **One-hot Encoding** – Labels are one-hot encoded for categorical cross-entropy loss.

---

## 🔹 Model Architecture

We use a **Sequential model** with the following layers:

1. **Embedding Layer** – Converts integer tokens to dense vectors of size 100.
2. **Bidirectional LSTM Layer** – 256 units, captures context from both directions.
3. **Dense Output Layer** – Softmax activation for predicting the next word.

```python
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(vocab_size, activation='softmax'))
```



Optimizer: Adam (learning rate = 0.01)
Loss function: Categorical Crossentropy
Metrics: Accuracy

🔹 Training

Batch size: 512

Epochs: 50 (with EarlyStopping callback monitoring training accuracy)

Early stopping stops training if there is less than 1% improvement in accuracy.

```Python
es = EarlyStopping(monitor='acc', min_delta=0.01)
model.fit(x, y, epochs=50, batch_size=512, callbacks=[es])
```

🔹 Text Generation

After training, the model can generate text given a seed text:
```Python
seed_text = "I could not help laughing at the ease with which he explained his process of deduction"
next_words = 100
```

The model predicts the next word iteratively.

The output is appended to the seed text to generate a story or passage.

🔹 Visualization

We track training accuracy and loss using Matplotlib:
```Python
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.show()
```
🔹 Key Takeaways

LSTM networks are effective for sequence prediction tasks like text generation.

Bidirectional LSTM captures context from past and future words.

Preprocessing (tokenization, sequence creation, padding) is crucial for NLP tasks.

Early stopping prevents overfitting and reduces training time.

🔹 Dependencies

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

🔹 References

Project Gutenberg: Sherlock Holmes

Keras Text Generation Tutorial
