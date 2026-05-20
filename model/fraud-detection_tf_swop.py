############################################################################
## (C)Copyright 2021-2023 Hewlett Packard Enterprise Development LP
############################################################################

import os
import json
import time
import numpy as np
import csv
import tensorflow as tf

from swarmlearning.tf import SwarmCallback


def load_dataset(file_path):

    with open(file_path, 'r') as f:

        data = np.array(
            list(csv.reader(f, delimiter=","))[1:],
            dtype=float
        )

    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1:]

    return X, y


# Constants
testFileName  = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

batchSize       = 32
defaultMaxEpoch = 100
defaultMinPeers = 2


def main():

    modelName = 'fraud-detection'

    dataDir = os.getenv(
        'DATA_DIR',
        '/platform/data'
    )

    scratchDir = os.getenv(
        'SCRATCH_DIR',
        '/platform/scratch'
    )

    resultFile = os.getenv(
        'RESULT_FILE',
        'results.json'
    )

    maxEpoch = int(
        os.getenv(
            'MAX_EPOCHS',
            str(defaultMaxEpoch)
        )
    )

    minPeers = int(
        os.getenv(
            'MIN_PEERS',
            str(defaultMinPeers)
        )
    )

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)

    print('-' * 64)

    # =========================
    # Load train dataset
    # =========================

    trainFile = os.path.join(
        dataDir,
        trainFileName
    )

    print(f"Loading train dataset {trainFile} ..")

    x_train, y_train = load_dataset(trainFile)

    print(
        f"Size of training dataset: "
        f"{len(x_train)}"
    )

    print('-' * 64)

    # =========================
    # Load test dataset
    # =========================

    testFile = os.path.join(
        dataDir,
        testFileName
    )

    print(f"Loading test dataset {testFile} ..")

    x_test, y_test = load_dataset(testFile)

    print(
        f"Size of test dataset: "
        f"{len(x_test)}"
    )

    print('-' * 64)

    # =========================
    # TF Dataset pipelines
    # =========================

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    )

    train_ds = train_ds.shuffle(
        len(x_train)
    )

    train_ds = train_ds.batch(
        batchSize,
        drop_remainder=True
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    )

    val_ds = val_ds.batch(batchSize)

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    # =========================
    # Model
    # =========================

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Dense(
            1,
            input_shape=(30,),
            activation='sigmoid',
            kernel_initializer='random_uniform',
            bias_initializer='zeros'
        )
    )

    sgd = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        decay=1e-6,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=[
            tf.keras.metrics.AUC(
                curve='ROC',
                name='auc_roc'
            ),
            tf.keras.metrics.AUC(
                curve='PR',
                name='auc_pr'
            )
        ]
    )

    print(model.summary())

    # =========================
    # Swarm callback
    # =========================

    swarmCallback = SwarmCallback(
        syncFrequency=128,
        minPeers=minPeers,
        adsValData=val_ds,
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        totalEpochs=maxEpoch
    )

    # =========================
    # Training
    # =========================

    print('Starting training ...')

    train_start = time.time()

    model.fit(
        train_ds,
        epochs=maxEpoch,
        validation_data=val_ds,
        callbacks=[swarmCallback]
    )

    train_end = time.time()

    training_time = round(
        train_end - train_start,
        2
    )

    print('Training done!')

    print(
        f"***** Training time: "
        f"{training_time}s "
        f"({round(training_time / 60, 2)} min)"
    )

    # =========================
    # Evaluation
    # =========================

    scores = model.evaluate(
        val_ds,
        verbose=1
    )

    print('***** Test loss:', scores[0])

    print('***** Test auc_roc:', scores[1])

    print('***** Test auc_pr:', scores[2])

    # =========================
    # Save Results
    # =========================

    results = {
        "config": {
            "dp_enabled": False,
            "noise_multiplier": None,
            "l2_norm_clip": None,
            "microbatches": None,
            "optimizer": "sgd",
            "learning_rate": "default",
            "metric": "both",
            "epochs": maxEpoch
        },

        "privacy": {
            "epsilon": None,
            "delta": None
        },

        "results": {
            "loss": round(float(scores[0]), 4),
            "auc_roc": round(float(scores[1]), 4),
            "auc_pr": round(float(scores[2]), 4)
        },

        "timing": {
            "training_time_seconds": training_time,
            "training_time_minutes": round(
                training_time / 60,
                2
            )
        }
    }

    results_path = os.path.join(
        "/results",
        resultFile
    )

    with open(results_path, 'w') as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(f"Results saved to {results_path}")

    # =========================
    # Save model
    # =========================

    model_path = os.path.join(
        scratchDir,
        modelName
    )

    model.save(model_path)

    print('Saved the trained model!')


if __name__ == '__main__':
    main()
