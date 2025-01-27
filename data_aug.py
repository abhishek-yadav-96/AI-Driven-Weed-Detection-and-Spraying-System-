import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image

# Define the input and output directories
input_dir = 'type 2'  # Folder containing your dataset of images
output_dir = 'augmented_images 2'  # Folder to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Initialize the ImageDataGenerator class with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5, 1.5)
)

# Parameters
image_size = (224, 224)  # Resize all images to this size
num_augmented_images = 5  # Number of augmented images to generate per original image

# Loop through all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(input_dir, filename)
        
        # Load and preprocess the image
        img = load_img(img_path)
        img = img.resize(image_size, Image.Resampling.LANCZOS)  # Resize the image
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # Reshape the image

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(
            x, 
            batch_size=1, 
            save_to_dir=output_dir, 
            save_prefix=os.path.splitext(filename)[0], 
            save_format='jpeg'
        ):
            i += 1
            if i >= num_augmented_images:
                break

print(f"Augmented images are saved in '{output_dir}' directory.")
