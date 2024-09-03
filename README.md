# CNN Model Implementation Using TensorFlow/Keras

This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) for binary classification tasks using TensorFlow/Keras. The notebook provides an end-to-end pipeline, including data preprocessing, model building, training, and evaluation.

## Contents

- **my_cnn Class**: A custom class that encapsulates the CNN architecture and its methods for building, training, and evaluating the model.
- **Data Preprocessing**: Code snippets for reading and cleaning the data using pandas.
- **Model Training**: Implementation of the training process using the Keras `fit` function, along with validation.
- **Model Evaluation**: Methods to evaluate the model's performance on test data.

## Dependencies

To run the notebook, you need the following Python libraries:

- TensorFlow
- Keras (included with TensorFlow)
- Pandas
- NumPy

You can install the necessary packages using:

```bash
pip install tensorflow pandas numpy
```

## Usage

### 1. Data Preparation

Ensure your dataset is available in CSV format. The data should be preprocessed, with numeric columns ready for input into the CNN model.

### 2. Model Initialization

The CNN model is initialized using the `my_cnn` class. The constructor takes the following parameters:

- `input_size`: The input shape of the data.
- `optimizer`: The optimizer to be used (default is `'adam'`).
- `loss`: The loss function (default is `'binary_crossentropy'`).
- `num_classes`: The number of output classes (default is `2` for binary classification).
- `embedding_size`: The size of the embedding layer (default is `128`).

### 3. Model Training

To train the model, use the `train` method, which takes:

- `X_train`: Training data features.
- `y_train`: Training data labels.
- `X_val`: Validation data features.
- `y_val`: Validation data labels.
- `epochs`: Number of training epochs.
- `batch_size`: Size of each training batch.

Example:
```python
model = my_cnn(input_size=(100, 1), optimizer='adam', loss='binary_crossentropy', num_classes=2)
model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
```

### 4. Model Evaluation

After training, evaluate the model on the test data using the `evaluate` method.

### 5. Predictions

Generate predictions using the `predict` method, which returns the model's predictions on new data.

## Contributing

If you find any issues or have suggestions for improvement, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This `README.md` file should provide a clear overview of the notebook's purpose, usage, and the key elements involved in the CNN model implementation. Let me know if there are any specific details you would like to add or modify!
