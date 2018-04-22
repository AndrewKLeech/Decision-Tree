import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Max depth for decision tree
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 10

# All column names for data set
col_names = ['id', 'age', 'job', 'marital', 'education',
             'default', 'balance', 'housing', 'loan', 'contact',
             'day', 'month', 'duration', 'campaign', 'pdays',
             'previous', 'poutcome', 'subscribed']

# Get training data and set the index to be the id column
trainingData = pd.read_csv('data/trainingset.txt', header=None, index_col=0, names=col_names)

# Get query data and set the index to be the id column
queryData = pd.read_csv('data/queries.txt', header=None, index_col=0, names=col_names)

# Create data frames
trainingDataFrame = pd.DataFrame(data=trainingData)
queryDataFrame = pd.DataFrame(data=queryData)

# Remove contact and duration column from each data frame as they have a lot of missing values
del trainingDataFrame['contact']
del trainingDataFrame['duration']
del queryDataFrame['contact']
del queryDataFrame['duration']

# Define target feature
targetLabels = trainingDataFrame['subscribed']

# Define continuous features
numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
numeric_dfs = trainingDataFrame[numeric_features]
numeric_dfs.head()

# Define categorical features
cat_dfs = trainingDataFrame.drop(numeric_features + ['subscribed'], axis=1)

# Handle missing values
cat_dfs.replace('unknown','NA')
cat_dfs.fillna( 'NA', inplace = True )

# Transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()

# Convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

encoding_dictionary = vectorizer.vocabulary_
for k in sorted(encoding_dictionary.keys()):
    mapping = k + " : column " + str(encoding_dictionary[k]) + " = 1"

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

# Set the model parameters to use entropy based information gain and have a max depth of MAX_DEPTH
decTreeModel = tree.DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

# Fit the model using the numeric representations of the training data
decTreeModel.fit(train_dfs, targetLabels)


# Define the continuous features from query data
q_num = queryDataFrame[numeric_features].as_matrix()

# Define the categorical features
q_cat = queryDataFrame.drop(numeric_features + ['subscribed'], axis=1)

# Convert categorical features
q_cat_dfs = q_cat.T.to_dict().values()
q_vec_dfs = vectorizer.transform(q_cat_dfs)

# Merge continuous and categorical features
query = np.hstack((q_num, q_vec_dfs))

# Get predictions for queries
predictions = decTreeModel.predict(query)

# Create output to write to file
output = ""
dataFrameLne = len(queryDataFrame.index)
for i in range(dataFrameLne):
    output += (queryDataFrame.index[i] + ",\"" + predictions[i] + "\"")
    if i < dataFrameLne - 1:
        output += "\n"


# Write to file
text_file = open("data/predictions.txt", "w")
text_file.write("%s" % output)
text_file.close()

# Hold-out Test Set + Confusion Matrix
print("-------------------------------------------------")
print("Accuracy and Confusion Matrix on Hold-out Testset")
print("-------------------------------------------------")

# Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

# Fit the model using just the test set
decTreeModel.fit(instances_train, target_train)

# Use the model to make predictions for the test set queries
predictions = decTreeModel.predict(instances_test)

# Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

# Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Cross-validation
print("------------------------")
print("Cross-validation Results")
print("------------------------")

# Run a 10 fold cross validation on this model using
scores=cross_validation.cross_val_score(decTreeModel, instances_train, target_train, cv=10)

# The cross validaton function returns an accuracy score for each fold
print("Entropy based Model:")
print("Score by fold: " + str(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")