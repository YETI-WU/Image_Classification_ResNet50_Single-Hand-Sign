# Single-Hand-Sign_Image_Classification_with_ResNet50  
Image Classification of Single-Hand-Sign
  
## Demon Image  
Randomly pick a number and demo the image & class. 
  
index = np.random.randint(len(Y_train_orig[0])) # pick a random integer from max/length number of train data  
print("index = " + str(index) + " ; " + "class = " + str(Y_train_orig[0,index]))  
plt.imshow(X_train_orig[index])  
  
![](images/DemoIamge_index337class2.png)
index = 337 ; class = 2 
  
  
## Convert Class to One Hot Encoder  
![](images/sign_class.png)  
  
  
## ResNet50  
Use Deep Neural Network model "ResNet50" to train the data.  
For ResNet50 coding, please see my previous repository.  
https://github.com/YETI-WU/Residual-Network  
  
  
## Plot Loss and Accuracy 
![](images/)  
  
  
## Test Model

