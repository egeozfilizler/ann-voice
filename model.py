import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load data from a CSV file and display basic information."""
    df = pd.read_csv(file_path)
    print(df.head())
    print(f'Number of Rows : {df.shape[0]}')
    print(f'Number of Columns : {df.shape[1]}')
    print(df.describe())
    return df


def prepare_features_and_labels(df):
    """
    Extract features and labels from dataframe.
    Features: first 20 columns (audio features)
    Labels: column 21 (gender: male/female)
    """
    # Get first 20 columns as input
    X = df.iloc[:, 0:20].values.astype(np.float32)
    Y = df.iloc[:, 20].values
    
    # Convert labels to numeric (male=1, female=0)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y).astype(np.float32)
    
    return X, Y, label_encoder


def split_train_test(X, Y, test_size=0.20, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    print(f'Shape of Train Data : {X_train.shape}')
    print(f'Shape of Test Data : {X_test.shape}')
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Normalize features using StandardScaler.
    Fit on training data and transform both train and test data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures normalized using StandardScaler")
    print(f"Mean of scaled training data: {np.mean(X_train_scaled, axis=0)[:3]}...")
    print(f"Std of scaled training data: {np.std(X_train_scaled, axis=0)[:3]}...")
    
    return X_train_scaled, X_test_scaled, scaler


def define_model(input_dim=20):
    """Define and compile the neural network model."""
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'), 
        Dense(16, activation='relu'),                       # Hidden layer                    
        Dense(1, activation='sigmoid'),                     # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=65, batch_size=8, verbose=0):
    """Train the model with training data and validate with test data."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return history


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model on both training and test data and print accuracies."""
    # Evaluate on training data
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    print(f'Training Accuracy: {round(train_scores[1]*100, 2)}%')
    print(f'Training Loss: {round(train_scores[0], 4)}')
    
    # Evaluate on test data
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {round(test_scores[1]*100, 2)}%')
    print(f'Test Loss: {round(test_scores[0], 4)}')
    
    return train_scores, test_scores


def plot_training_history(history):
    """Plot training and validation accuracy over epochs."""
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def save_model(model, filepath='my_model.keras'):
    """Save the trained model to a file."""
    model.save(filepath)
    print(f'\nModel saved to {filepath}')


def ann_keras():
    """Main function to orchestrate the entire ML pipeline."""
    # Load data
    df = load_data('assets/data.csv')
    
    # Prepare features and labels
    X, Y, label_encoder = prepare_features_and_labels(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, Y)
    
    # Normalize features using StandardScaler
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Define the model
    model = define_model(input_dim=20)
    
    # Train the model with scaled data
    print("\nTraining the model...")
    history = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=70, batch_size=16, verbose=0)
    
    # Evaluate the model on both training and test data
    print("\nEvaluating the model...")
    evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save the model
    save_model(model, 'my_model.keras')
    
    # Plot training history
    plot_training_history(history)
    
    return model, X_test_scaled, y_test, label_encoder, scaler


def main():
    """Entry point of the program."""
    ann_keras()


if __name__ == "__main__":
    main()