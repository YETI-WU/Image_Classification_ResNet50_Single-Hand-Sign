# model_Test.py
"""
Test the trained model using HDF5 file 'test_signs.h5' in datasets.
"""

# Test model
scores = model.evaluate(X_test, Y_test)
print ("Loss = " + str(scores[0]))
print ("Test Accuracy = " + str(scores[1]))

