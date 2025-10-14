import pandas as pd
import numpy as np

def ann_keras():

    # Get the data
    df = pd.read_csv('voice.csv')
    print(df.head())

    # Data information
    print ('Number of Rows :', df.shape[0])
    print ('Number of Columns :', df.shape[1])
    print(df.describe())

    # Get first 20 columns as input
    X = df.iloc[:, 0:20].values.astype(np.float32) # Inputs - convert to float32
    Y = df.iloc[:, 20].values # Outputs
    
    # Convert labels to numeric (male=1, female=0)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y).astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

    # Data shapes
    print (f'Shape of Train Data : {X_train.shape}')
    print (f'Shape of Test Data : {X_test.shape}')

    # Define the model
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential([
        Dense(32, input_dim = (20), activation = 'relu'), # Input layer
        Dense(16, activation = 'relu'),
        Dense(1, activation = 'sigmoid'), # Output layer
    ])

    # Define number of epochs
    epoch_n = 120

    # Compile the model
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epoch_n, batch_size=16, verbose = 0)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test)
    print(model.metrics_names)
    print(f'Accuracy : {model.metrics_names[1]} : {round(scores[1]*100,2)}%')

    # Plot training & validation accuracy values
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def main():
    ann_keras()

if __name__ == "__main__":
    main()
