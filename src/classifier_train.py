import sys
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from get_features import get_features
from sklearn.model_selection import train_test_split


# CSV file with image paths and corresponding labels (e.g., `data/test.csv`)
f_path = sys.argv[1]


# Retrieve dataset and save to `X` (HoG features) and `y` (binary label)
data = pd.read_csv(f_path, header=None, dtype={1: np.bool_})
n = len(data.index)

X = np.zeros((n,1188))
y = data.values[:,1].astype(np.bool_)
	
for i in range(n):   #range(n)
	img = cv2.imread(data.loc[i,0])[:,:,::-1]	
	X[i,:] = get_features(img)


# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
print("Number of training samples:", X_train.shape[0])
print("Number of training samples with True label:", np.count_nonzero(y_train))
print("Number of validation samples:", X_val.shape[0])
print("Number of validation samples with True label:", np.count_nonzero(y_val))


# Train linear classifier, report accuracies, and pickle classifier for later use
C = 1
clf = LinearSVC(random_state=0, C=C)
print("Training...")
clf.fit(X_train, y_train)
print("Training complete!")

print("Training accuracy:", clf.score(X_train, y_train))
print("Validation accuracy:", clf.score(X_val, y_val))

return_dict = {'classifier': clf}
outfile = open("pickled_classifier",'wb')
pickle.dump(return_dict, outfile)
outfile.close()
