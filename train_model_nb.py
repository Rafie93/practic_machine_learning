import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


bankdata = pd.read_csv('dataset-nasabah.csv')

print(bankdata)

X = bankdata.drop(['ID_Nasabah','JenisKelamin', 'Class'], axis=1)
Y = bankdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)

# Train the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Make prediction on the test set
y_predict = clf.predict(X_test)
# print(y_predict)

accuracy = accuracy_score(y_test,y_predict)
cm = confusion_matrix(y_test,y_predict,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()

print("Hasil Akurasi " + str(accuracy))


# Save model
with open('model-knn.pickle', 'wb') as f:
    pickle.dump(clf, f)

