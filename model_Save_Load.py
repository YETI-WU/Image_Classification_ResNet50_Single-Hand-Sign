# model_Save_Load
  
# Save model for future training
model.save('ResidualNet50_HE.h5') 

# Load model from previous trained model, and train again
#model = load_model('ResidualNet50.h5') 
#model.fit(X_train, Y_train, epochs = 1, batch_size = 32)

