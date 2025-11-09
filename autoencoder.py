import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1).values
y = data["Class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

input_dim = X.shape[1]
input_layer = keras.layers.Input(shape=(input_dim,))
encoded = keras.layers.Dense(128, activation='relu')(input_layer)
encoded = keras.layers.Dense(64, activation='relu')(encoded)
encoded = keras.layers.Dense(32, activation='relu')(encoded)
encoded = keras.layers.Dense(16, activation='relu')(encoded)
encoded = keras.layers.Dense(8, activation='relu')(encoded)
decoded = keras.layers.Dense(16, activation='relu')(encoded)
decoded = keras.layers.Dense(32, activation='relu')(decoded)
decoded = keras.layers.Dense(64, activation='relu')(decoded)
decoded = keras.layers.Dense(128, activation='relu')(decoded)
decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

X_train = X[y == 0]
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True, verbose=1)

plt.plot(history.history['loss'])
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.show()

reconstructed = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructed, 2), axis=1)

plt.figure(figsize=(8,4))
plt.hist(mse[y == 0], bins=100, alpha=0.6, label='Normal')
plt.hist(mse[y == 1], bins=100, alpha=0.6, label='Fraud')
plt.yscale('log')
plt.title('Reconstruction Error (Log Scale)')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency (log)')
plt.legend()
plt.show()
