import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def recognize():
    image_ds = tf.data.Dataset.from_tensors(
        'C:\\zimomo\\BigCreate\\XiaoMi\\1.8.0\\projects\\vs2017\\mynteyed_demo\\x64\\Release\\test.jpg').map(
        load_and_preprocess_image)
    # image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(load_and_preprocess_image)
    # 注意 from_tensor_slices 和 from_tensors 方法的区别
    test_image = np.array(list(image_ds.as_numpy_iterator()))
    model = keras.models.load_model(
        'C:\\zimomo\\BigCreate\\XiaoMi\\1.8.0\\projects\\vs2017\\mynteyed_demo\\x64\\Release\\saved_model')
    results = model.predict(test_image)
    result = np.argmax(results, axis=-1)
    return (str(result[0]), "temp")
