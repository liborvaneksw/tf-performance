import argparse
import os
import shutil
import tensorflow as tf
import numpy as np
import timeit
from enum import Enum, auto


class Format(Enum):
    SAVED_MODEL = auto()
    HDF5 = auto()
    WEIGHTS_SAVED_MODEL = auto()
    WEIGHTS_HDF5 = auto()
    TF_LITE = auto()


class Model(Enum):
    MOBILENET = auto()
    DENSENET = auto()
    RESNET = auto()


def create_model(model: Model):
    """
    Create model with imagenet weights, used for saving.
    """
    if model == Model.MOBILENET:
        return tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights="imagenet")

    if model == Model.RESNET:
        return tf.keras.applications.ResNet152V2(input_shape=(224, 224, 3), weights="imagenet")

    if model == Model.DENSENET:
        return tf.keras.applications.DenseNet169(input_shape=(224, 224, 3), weights="imagenet")


def create_model_with_weights(model: Model, weights_path: str):
    """
    Create model and load weights from given path. Used for measuring loading time.
    """
    model = create_model(model)
    model.load_weights(weights_path)
    return model


def load_tflite_model(directory: str):
    """
    Load TF lite model. Used for measuring loading time.
    """
    path = get_path(directory, Format.TF_LITE)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()


def get_path(directory: str, format: Format):
    """
    Creates folder / file name according to the given format type. (And uniqe to that type.)
    Models are saved in a separate folders.
    """
    path = os.path.join(directory, format.name.lower())

    if format == Format.HDF5 or format == Format.WEIGHTS_HDF5:
        return path + ".h5"

    if format == Format.TF_LITE:
        return path + ".tflite"

    return path


def get_params_count(model: tf.keras.Model):
    vars = model.variables
    return sum(map(lambda v: np.prod(v.shape), vars))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of tf.keras model loading time.")
    parser.add_argument(
        "--save_directory",
        type=str,
        default="temp",
        help="non existing directory name for saving models during execution",
    )
    parser.add_argument(
        "--repeat", type=int, default=20, help="how many times we should repeat the loading",
    )
    all_formats = ", ".join([f.name.lower() for f in list(Format)])
    parser.add_argument(
        "--formats", nargs="*", default=None, help=f"formats to compare ({all_formats}), default: all.",
    )
    args = parser.parse_args()

    try:
        print([f.upper() for f in args.formats])
        formats = (
            [Format[f.upper()] for f in args.formats] if args.formats is None or len(args.formats) > 0 else list(Format)
        )
    except ValueError as e:
        raise Exception(f"Attribute 'formats' can only contain following values: {all_formats}.") from e

    try:
        os.mkdir(args.save_directory)
    except Exception as e:
        raise Exception("Provide name of non existing directory inside the current directory.") from e

    try:
        # Prepare dictionaries we will fill later.
        param_count = {}
        loading_time = {format: {model: 0 for model in list(Model)} for format in formats}

        # First, save models with imagenet weights.
        for model in list(Model):
            tf_model = create_model(model)
            param_count[model] = get_params_count(tf_model)
            save_path = os.path.join(args.save_directory, model.name.lower())
            if Format.SAVED_MODEL in formats:
                tf_model.save(get_path(save_path, Format.SAVED_MODEL), save_format="tf", include_optimizer=False)
            if Format.HDF5 in formats:
                tf_model.save(get_path(save_path, Format.HDF5), save_format="h5", include_optimizer=False)
            if Format.WEIGHTS_SAVED_MODEL in formats:
                tf_model.save_weights(get_path(save_path, Format.WEIGHTS_SAVED_MODEL), save_format="tf")
            if Format.WEIGHTS_HDF5 in formats:
                tf_model.save_weights(get_path(save_path, Format.WEIGHTS_HDF5), save_format="h5")

            if Format.TF_LITE in formats:
                converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
                tflite_model = converter.convert()
                open(get_path(save_path, Format.TF_LITE), "wb").write(tflite_model)

        # Now, we can measure loading time of saved models.
        def measure_time(model, format, loading_function):
            if format not in formats:
                return

            start_time = timeit.default_timer()
            loading_function()
            end_time = timeit.default_timer()
            result = end_time - start_time
            loading_time[format][model] += result

            tf.keras.backend.clear_session()

        for i in range(args.repeat):
            for model in list(Model):
                save_path = os.path.join(args.save_directory, model.name.lower())
                measure_time(
                    model,
                    Format.SAVED_MODEL,
                    lambda: tf.keras.models.load_model(get_path(save_path, Format.SAVED_MODEL)),
                )

                measure_time(
                    model, Format.HDF5, lambda: tf.keras.models.load_model(get_path(save_path, Format.HDF5)),
                )

                measure_time(
                    model,
                    Format.WEIGHTS_SAVED_MODEL,
                    lambda: create_model_with_weights(model, get_path(save_path, Format.WEIGHTS_SAVED_MODEL)),
                )

                measure_time(
                    model,
                    Format.WEIGHTS_HDF5,
                    lambda: create_model_with_weights(model, get_path(save_path, Format.WEIGHTS_HDF5)),
                )

                measure_time(
                    model, Format.TF_LITE, lambda: load_tflite_model(save_path),
                )
    finally:
        shutil.rmtree(args.save_directory)
        pass

    # Print as as coma separated values. Columns: model name, parameters, loading times for selected formats.
    print(",parameters," + ",".join([f.name for f in formats]))
    for model in list(Model):
        avg_times = [(loading_time[f][model] / args.repeat) for f in formats]
        avg_times_string = ["{:.3f}".format(t) for t in avg_times]
        print(model.name + "," + str(param_count[model]) + "," + ",".join(avg_times_string))
