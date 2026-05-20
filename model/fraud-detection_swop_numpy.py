############################################################################
## (C)Copyright 2021-2023 Hewlett Packard Enterprise Development LP
############################################################################

import os
import json
import time
import numpy as np
import csv
import logging
import tensorflow as tf

from swarmlearning.tf import SwarmCallback


def getXY(dataSet):
    np.random.shuffle(dataSet)

    length = np.size(dataSet,0)

    X = dataSet[0:length, :-1]
    y = dataSet[0:length, -1:]

    return X , y


# Constants
testFileName = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

part = 0
batchSize = 32

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

    # ================== load test and train Data =========================

    print('-' * 64)

    trainFile = dataDir + '/' + trainFileName

    print("loading train dataset %s .." % trainFile)

    with open(trainFile, 'r') as f:

        trainData = np.array(
            list(csv.reader(f, delimiter=","))[1:],
            dtype=float
        )

        print(
            'size of training Data set : %s'
            % np.size(trainData,0)
        )

    print('-' * 64)

    testFile = dataDir + '/' + testFileName

    print("loading test dataset %s .." % testFile)

    with open(testFile, 'r') as f:

        testData = np.array(
            list(csv.reader(f, delimiter=","))[1:],
            dtype=float
        )

        print(
            'size of test Data set : %s'
            % np.size(testData,0)
        )

    print('-' * 64)

    # ================== Model =========================

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

    print('Starting training ...')

    x_train, y_train = getXY(trainData)

    x_test, y_test = getXY(testData)

    # ================== Swarm callback =========================

    swarmCallback = SwarmCallback(
        syncFrequency=128,
        minPeers=minPeers,
        adsValData=(x_test, y_test),
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        totalEpochs=maxEpoch
    )

    # ================== Train =========================

    train_start = time.time()

    model.fit(
        x_train,
        y_train,
        batch_size=batchSize,
        epochs=maxEpoch,
        validation_data=(x_test, y_test),
        shuffle=True,
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

    # ================== Evaluate =========================

    scores = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print('***** Test loss:', scores[0])

    print('***** Test auc_roc:', scores[1])

    print('***** Test auc_pr:', scores[2])

    # ================== Save Results =========================

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

    # ================== Save Model =========================

    model_path = os.path.join(
        scratchDir,
        modelName
    )

    model.save(model_path)

    print('Saved the trained model!')


if __name__ == '__main__':
    main()
