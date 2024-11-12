import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


### Step 2: Load and Preprocess Data

# Load your car front images dataset here (normalized to the range [-1, 1] for GAN training)
# Assuming `car_images` is an array of shape (num_images, img_height, img_width, channels)
car_images = np.load("C:/Users/asus/volkswagan/car images")

BUFFER_SIZE = car_images.shape[0]
BATCH_SIZE = 32

# Prepare the data pipeline
train_dataset = tf.data.Dataset.from_tensor_slices(car_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


### Step 3: Build the Generator Network

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = build_generator()


### Step 4: Build the Discriminator Network

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

discriminator = build_discriminator()


### Step 5: Define Loss Functions and Optimizers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


### Step 6: Define Training Step

EPOCHS = 50
noise_dim = 100

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


### Step 7: Train the Model


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        # Generate and save images for each epoch

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, tf.random.normal([1, noise_dim]))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(1, 1, i+1)
        plt.imshow((predictions[i] * 127.5 + 127.5).numpy().astype(int))
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

train(train_dataset, EPOCHS)


### Step 8: Generate New Designs

# Generate a new design
new_design = generator(tf.random.normal([1, noise_dim]), training=False)

plt.imshow((new_design[0] * 127.5 + 127.5).numpy().astype(int))
plt.axis('off')
plt.show()