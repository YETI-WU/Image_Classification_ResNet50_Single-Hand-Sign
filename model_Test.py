# model_Test.py

# Test model
scores = model.evaluate(X_test, Y_test)
print ("Loss = " + str(scores[0]))
print ("Test Accuracy = " + str(scores[1]))

