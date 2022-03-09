# Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Read data into a dataframe
dataset = pd.read_csv("life_expectancy.csv")

# Examine five rows of dataset
print(dataset.head())

# Print summary statistics of dataset
print(dataset.describe())

# Drop the Country column from dataset
dataset = dataset.drop(['Country'], axis=1)

# Create labels (Contained in the "Life expectancy") column
labels = dataset.iloc[:, -1]

# Create Features dataset
features = dataset.iloc[:, 0:-1]

# Convert the features categorical column to a numerical column 
features = pd.get_dummies(features)

# Split data into training set and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=30)

# Normalizing the data
ct = ColumnTransformer([
  (
    "only numeric",
    StandardScaler(),
    features.select_dtypes(include=["float64", "int64"]).columns
  )
], remainder="passthrough")

# Fit and transform ct to the training data
features_train_scaled = ct.fit_transform(features_train)

# Transform test data instance
features_test_scaled = ct.transform(features_test)

# Create a neural network model
my_model = Sequential()

# Create an input layer to the model
input = InputLayer(input_shape = (features.shape[1], ))

# Add the input layer to the model (my_model)
my_model.add(input)

# Add hidden layers to the model
my_model.add(Dense(64, activation="relu"))

# Add an ouput layer with one neuron to the model
my_model.add(Dense(1))

# Print out a summary statistics of the model (my_model)
print(my_model.summary())

# Create an instance of Adam optimizer
opt = Adam(learning_rate=0.01)

# Compile the model
my_model.compile(
  loss="mse",
  metrics=["mae"],
  optimizer=opt
)

# Train the model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Evaluate the trained model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

# Print out the final loss and the final metric
print(res_mse, res_mae)