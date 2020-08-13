"""Utility functions used in notebooks
Changes are made upon Krzysztof Kolasinski's script to adapt to the project.
"""


from collections import defaultdict
from typing import Optional, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_gan as tfgan


def numpy_array_to_dataset(
        array: np.array,
        buffer_size: int = 512,
        batch_size: int = 100,
        num_parallel_batches: int = 16,
        preprocess_fn: Optional[Callable] = None,
) -> tf.data.Dataset:
    """Convert numpy array to tf.data.Dataset"""
    dataset = tf.data.Dataset.from_tensor_slices(array.astype(np.float32))
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=-1)
    )
    if preprocess_fn is None:
        dataset = dataset.batch(batch_size=batch_size)
    else:
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=preprocess_fn,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches,
                drop_remainder=True,
            )
        )
    dataset = dataset.prefetch(4)
    return dataset


def create_tfrecord_dataset_iterator(
        tfrecord_paths: List[str],
        image_size: int = 256,
        batch_size: int = 8,
        buffer_size: int = 512,
        num_parallel_batches: int = 16,
) -> tf.Tensor:
    """
    Dataset loader
    Args:
        tfrecord_paths: path to the tfrecords with stacked images from mat_to_jpg.py
        image_size: a resize size of the input images by CROPPING NOT DOWNSAMPLING
        batch_size: batch size
        buffer_size: shuffle buffer size
        num_parallel_batches: number of parallel calls when reading
            and preparing dataset

    Returns:
        image a tensor iterator of shape [batch_size, image_size, image_size, 3]
        also IT MUST BE NOTED THAT this function converts dtype=UINT8 into float 32, reason will be shown in jpeg_to_tensor.ipynb in main_code
    """
    dataset = tf.data.TFRecordDataset(tfrecord_paths)


    def preprocess_images(example):
        features = tf.parse_single_example(
            example, features={"image_raw": tf.FixedLenFeature([], tf.string)}
        )

        image = tf.decode_raw(features["image_raw"], tf.uint8)
        image.set_shape([256 * 256 * 3])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [256, 256, 3])
        # Taking top-left subimage (Part 1) for example, notice the origin of the coordinates is the top-left vertex
        image = image[0:128, 0:128, :]
        image = tf.reshape(image, [128, 128, 3])
        return image

    if buffer_size > 0:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size,
                                               count=-1)
        )

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=preprocess_images,
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=True,
        )
    )
    dataset = dataset.prefetch(4)
    images = dataset.make_one_shot_iterator().get_next()


    x_in = tf.reshape(images, [batch_size, 128, 128, 3])
    x_in = tf.image.resize_images(
        x_in, [image_size, image_size], method=0, align_corners=False
    )
    return x_in / 255.0


_epsilon = 1e-5


def safe_log(x: tf.Tensor) -> tf.Tensor:
    return tf.log(tf.maximum(x, _epsilon))


class Metrics:
    def __init__(self, step, metrics_tensors):
        self.metrics = defaultdict(list)
        self.step = step
        self.metrics_tensors = metrics_tensors

    def check_step(self, i):
        return (i + 1) % self.step == 0

    def append(self, results):
        for k, t in self.metrics_tensors.items():
            self.metrics[k].append(results[k])
            print(k, results[k])

    def get(self):
        return self.metrics_tensors

    @property
    def num_metrics(self):
        return len(self.metrics)


class PlotMetricsHook:
    def __init__(self, metrics: Metrics, step=1000, figsize=(15, 3),
                 skip_steps=5):
        self.metrics = metrics
        self.step = step
        self.figsize = figsize
        self.skip_steps = skip_steps

    def check_step(self, i):
        return (i + 1) % self.step == 0

    def run(self):
        plt.figure(figsize=self.figsize)

        for k, (m, values) in enumerate(self.metrics.metrics.items()):
            plt.subplot(1, self.metrics.num_metrics, k + 1)
            plt.title(m)
            vals = values[self.skip_steps:]
            plt.plot(vals)
            vals = np.array(vals)
            if len(vals) > 0:
                plt.ylim([vals.min(), vals.max()])
        plt.show()


def trainer(sess, num_steps, train_op, feed_dict_fn, metrics, hooks):
    for i in tqdm(range(num_steps)):
        fetches = {"train_op": train_op}

        for metric in metrics:
            if metric.check_step(i):
                fetches.update(**metric.get())

        results = sess.run(fetches=fetches, feed_dict=feed_dict_fn())

        for metric in metrics:
            if metric.check_step(i):
                metric.append(results)

        for hook in hooks:
            if hook.check_step(i):
                hook.run()


def plot_4x4_grid(
        images: np.ndarray,
        shape: tuple = (28, 28),
        cmap="gray",
        figsize=(4, 4)
) -> None:

    assert images.shape[0] >= 16
    dist_samples_np = images[:16, ...].reshape([4, 4, *shape])

    plt.figure(figsize=figsize)
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow(dist_samples_np[i, j], cmap=cmap)
            plt.xticks([])
            plt.yticks([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)


def plot_grid(images: tf.Tensor) -> tf.Tensor:
    """
    Plot grid of images. Be careful with the version of tfgan & gast, which could easily go wrong
    Args:
        images: a tensor with batch of images of shape
            [batch_size, size, size, 3]

    Returns:
        a grid image
    """
    batch_size, image_size = images.shape.as_list()[:2]

    grid_image = tfgan.eval.image_grid(
        images,
        grid_shape=[4, batch_size // 4],
        image_shape=(image_size, image_size),
        num_channels=3
    )

    return grid_image[0]
