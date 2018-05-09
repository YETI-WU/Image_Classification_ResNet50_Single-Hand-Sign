# model_Build_Compile.py

# Build model graph
model = ResNet50(input_shape = (64, 64, 3), classes = 6)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

