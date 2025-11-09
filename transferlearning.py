from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load dataset (auto splits into train and test)
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = data_gen.flow_from_directory('caltech-101-img', target_size=(224,224), batch_size=32, subset='training')
test = data_gen.flow_from_directory('caltech-101-img', target_size=(224,224), batch_size=32, subset='validation')

# a. Load pretrained VGG16 (using your .h5 weights file)
base = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
             include_top=False, 
             input_shape=(224,224,3)) 

# b. Freeze base layers
for layer in base.layers:
    layer.trainable = False

# c. Add custom classifier
model = Sequential([
    base,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

# d. Train classifier
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=5, validation_data=test)

# e. Fine-tune last few layers
for layer in base.layers[-4:]:
    layer.trainable = True
model.fit(train, epochs=2, validation_data=test)