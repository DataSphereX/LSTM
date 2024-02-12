# LSTM
Long Short-Term Memory: The RNN Hero Conquering Long Sequences

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode the target variable
enc = OneHotEncoder(sparse=False)
y_encoded = enc.fit_transform(y.reshape(-1, 1))

# Reshape X for LSTM input
X_lstm = X.reshape(X.shape[0], 1, X.shape[1])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_encoded, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("LSTM Accuracy:", accuracy)

