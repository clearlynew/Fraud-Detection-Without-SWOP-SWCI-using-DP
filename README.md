# Fraud Detection with Swarm Learning + Differential Privacy

## Overview

This project demonstrates distributed fraud detection training using:

* HPE Swarm Learning
* TensorFlow/Keras
* Differential Privacy (DP-SGD / DP-Adam)
* TensorFlow Privacy

The implementation supports:

* Standard SGD / Adam training
* Differentially Private SGD and Adam
* Configurable Gaussian noise
* Privacy accounting (Epsilon & Delta reporting)
* AUC-ROC and AUC-PR metrics
* Automatic JSON result saving
* Automatic ML log collection

---

# Project Structure

```text
fraud-detection/
├── cert/
├── data-and-scratch1/
├── data-and-scratch2/
├── ml-context/
├── model/
│   └── fraud-detection.py
├── results/
│   ├── *.json
│   └── *.log
├── tmp/
│   ├── sl1/
│   └── sl2/
└── README.md
```

---

# 1. Clone Project Repository into Workspace

```bash
cd ~/swarm-learning/workspace/

git clone https://github.com/clearlynew/Fraud-Detection-Without-SWOP-SWCI-using-DP.git fraud-detection
```

---

# 2. Generate Certificates

```bash
cd ~/swarm-learning/

cp -r examples/utils/gen-cert workspace/fraud-detection/

./workspace/fraud-detection/gen-cert -e fraud-detection -i 1

./workspace/fraud-detection/gen-cert -e fraud-detection -i 2
```

---

# 3. Delete certificates with "swop" and "swci" in their name

```bash
cd workspace/fraud-detection/cert

rm swop-* swci-*

cd ../../../
```

---

# 4. Create Docker Network

```bash
docker network create host-1-net
```

---

# 5. Create Required Directories

```bash
mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl1

mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl2

mkdir -p ~/swarm-learning/workspace/fraud-detection/results

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/tmp

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/results
```

---

# 6. Copy SwarmLearning Wheel and Remove Duplicate

```bash
cp ~/swarm-learning/lib/swarmlearning-*.whl \
~/swarm-learning/workspace/fraud-detection/ml-context/

rm workspace/fraud-detection/ml-context/swarmlearning-client-*.whl 2>/dev/null
```

---

# 7. Build ML Docker Image

```bash
docker build -t fraud-ml-env ~/swarm-learning/workspace/fraud-detection/ml-context
```

---

# 8. Run APLS

```bash
docker run -d \
--name apls \
--network host-1-net \
-v apls-volume:/hpe \
-p 5814:5814 \
--restart unless-stopped \
hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19
```

---

# Set Environment Variables

```bash
export HOST_IP=172.1.1.1

export SN_IP=172.1.1.1

export APLS_IP=172.1.1.1

export SN_API_PORT=30304
```

---

# 9. Run SN (Swarm Network Node)

```bash
cd ~/swarm-learning

./scripts/bin/run-sn -d --name=sn1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sentinel \
--sn-api-port=${SN_API_PORT} \
--key=workspace/fraud-detection/cert/sn-1-key.pem \
--cert=workspace/fraud-detection/cert/sn-1-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
--apls-ip=${APLS_IP}
```

---

# 10. Monitor SN Until Ready

```bash
docker logs -f sn1
```

Wait until:

```text
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

---

# Between Experiments

Stop old containers before every new experiment:

```bash
docker rm -f sn1 sl1 sl2 ml1 ml2 2>/dev/null
```

---

# Experiment Summary (With and Without SWOP)

| Exp | DP  | Noise | Epochs | Optimizer | 
| --- | --- | ----- | ------ | --------- |
| 1   | No  | 0.0   | 8      | SGD       |
| 2   | Yes | 0.1   | 8      | SGD       |
| 3   | Yes | 0.5   | 8      | SGD       |
| 4   | Yes | 1.0   | 8      | SGD       |
| 5   | Yes | 3.0   | 8      | SGD       |

---

# Experiment 1 — Baseline

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fraud-detection/cert/sl-1-key.pem \
--cert=workspace/fraud-detection/cert/sl-1-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
--ml-image=fraud-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fraud-detection.py \
-v ~/swarm-learning/workspace/fraud-detection/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch1/app-data:/app-data \
--ml-v workspace/fraud-detection/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp1_baseline_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=sgd \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > ~/swarm-learning/workspace/fraud-detection/results/exp1_baseline_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fraud-detection/cert/sl-2-key.pem \
--cert=workspace/fraud-detection/cert/sl-2-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
--ml-image=fraud-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fraud-detection.py \
-v ~/swarm-learning/workspace/fraud-detection/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch2/app-data:/app-data \
--ml-v workspace/fraud-detection/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp1_baseline_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=sgd \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > ~/swarm-learning/workspace/fraud-detection/results/exp1_baseline_ml2.log 2>&1 &
```

---

# Experiments 2–10

Use the same commands while changing:

* `RESULT_FILE`
* `NOISE_MULTIPLIER`
* `MAX_EPOCHS`
* `OPTIMIZER`

according to the experiment table above.

---

# Results Storage

Each experiment automatically saves:

* JSON metrics/results
* Full ML training logs

inside:

```text
workspace/fraud-detection/results/
```

---

# Running Fraud Detection Using SWOP

## 1. Configure SWCI for 2 Peers

Edit the SWCI initialization file and change the setup from 4 peers to 2 peers.

---

## 2. Update the SWOP Profile

Remove two node definitions and keep only 2 nodes.

Add `RESULT_FILE` inside `userEnvars` for both node definitions.

Example:

```yaml id="mxz1jd"
userEnvars:
[
    SCRATCH_DIR : "user1",
    RESULT_FILE : "expt2_swop_tf_ml1_results.json"
]
```

Second node:

```yaml id="h5if81"
userEnvars:
[
    SCRATCH_DIR : "user2",
    RESULT_FILE : "expt2_swop_tf_ml2_results.json"
]
```

---

## 3. Mount the Results Folder

Go to:

```text id="g2y2xv"
swci/taskdefs/swarm_fd/task.yaml
```

Add:

```yaml id="e4m4jm"
- Src: /path/to/your/fraud-detection/results
  Tgt: /results
  MType: BIND
```

Example:

```yaml id="jpb6dr"
- Src: /home/<your-username>/swarm-learning/workspace/fraud-detection/results
  Tgt: /results
  MType: BIND
```

---

## 4. Set Minimum Peers and Epochs

In:

```text id="ibpx4f"
swci/taskdefs/swarm_fd/task.yaml
```

set:

```yaml id="a2um2i"
"MAX_EPOCHS": 8,
"MIN_PEERS": 2
```

---

## 5. Use the Required Fraud Detection Script

Inside the `model` folder:

```text id="l4af8h"
model/
├── fraud-detection.py
├── fraud-detection_swop_numpy.py
└── fraud-detection_tf_swop.py
```

Use the version you want for the experiment and remove the rest from the configuration.

---

## 6. Run the Experiment

The results and logs will be saved inside:

```text id="8w9yiu"
fraud-detection/results/
```
---

# Differential Privacy Parameters

| Parameter          | Description                 |
| ------------------ | --------------------------- |
| `DP_ENABLED`       | Enables/disables DP         |
| `NOISE_MULTIPLIER` | Gaussian noise multiplier   |
| `L2_NORM_CLIP`     | Gradient clipping threshold |
| `MICROBATCHES`     | Number of microbatches      |
| `MAX_EPOCHS`       | Number of training epochs   |
| `OPTIMIZER`        | SGD or Adam                 |
| `METRIC`           | auc_roc / auc_pr / both     |
| `RESULT_FILE`      | Output JSON filename        |

---

# Expected Results

| Privacy ↑        | Utility ↓            |
| ---------------- | -------------------- |
| Stronger privacy | Lower AUC            |
| More noise       | Slower convergence   |
| Higher DP        | Longer training time |

---

# Notes

* TensorFlow version: `2.7.0`
* TensorFlow Privacy version: `0.7.3`
* TensorFlow Probability version: `0.15.0`

---

# Results (Hima)
  # ML1

  | SWOP              | DP Enabled | Noise Multiplier | Epsilon    | Delta     | Loss   | AUC-ROC | AUC-PR | Training Time (s) | Training Time (min) |
| ----------------- | ---------- | ---------------- | ---------- | --------- | ------ | ------- | ------ | ----------------- | ------------------- |
| Without SWOP      | No         | 0.0              | —          | —         | 1.0374 | 0.8700  | 0.8746 | 133.59            | 2.23                |
| Without SWOP      | Yes        | 0.1              | 34178.1677 | 0.0002857 | 0.6120 | 0.7647  | 0.8310 | 131.92            | 2.20                |
| Without SWOP      | Yes        | 0.5              | 10.6858    | 0.0002857 | 0.6251 | 0.7616  | 0.8257 | 134.05            | 2.23                |
| Without SWOP      | Yes        | 1.0              | 1.3626     | 0.0002857 | 0.6024 | 0.7706  | 0.8314 | 140.96            | 2.35                |
| Without SWOP      | Yes        | 3.0              | 0.2619     | 0.0002857 | 0.5874 | 0.7882  | 0.8452 | 133.94            | 2.23                |
| With SWOP (Numpy) | No         | 0.0              | —          | —         | 0.5523 | 0.8726  | 0.9130 | 162.12            | 2.70                |
| With SWOP (TF)    | No         | 0.0              | —          | —         | 0.5528 | 0.8705  | 0.9114 | 159.63            | 2.66                |
| With SWOP         | Yes        | 0.1              | 34178.1677 | 0.0002857 | 0.5603 | 0.8013  | 0.8582 | 164.77            | 2.75                |
| With SWOP         | Yes        | 0.5              | 10.6858    | 0.0002857 | 0.6189 | 0.7672  | 0.8286 | 164.43            | 2.74                |
| With SWOP         | Yes        | 1.0              | 1.3626     | 0.0002857 | 0.6004 | 0.7745  | 0.8369 | 161.51            | 2.69                |
| With SWOP         | Yes        | 3.0              | 0.2619     | 0.0002857 | 0.5480 | 0.8115  | 0.8670 | 162.58            | 2.71                |

---

 # ML2

| SWOP              | DP Enabled | Noise Multiplier | Epsilon    | Delta   | Loss   | AUC-ROC | AUC-PR | Training Time (s) | Training Time (min) |
| ----------------- | ---------- | ---------------- | ---------- | ------- | ------ | ------- | ------ | ----------------- | ------------------- |
| Without SWOP      | No         | 0.0              | —          | —       | 1.0597 | 0.8583  | 0.8506 | 137.59            | 2.29                |
| Without SWOP      | Yes        | 0.1              | 38389.3765 | 0.00025 | 0.6510 | 0.7419  | 0.7938 | 135.12            | 2.25                |
| Without SWOP      | Yes        | 0.5              | 10.3322    | 0.00025 | 0.6495 | 0.7399  | 0.7904 | 137.12            | 2.29                |
| Without SWOP      | Yes        | 1.0              | 1.2792     | 0.00025 | 0.6411 | 0.7502  | 0.7942 | 136.79            | 2.28                |
| Without SWOP      | Yes        | 3.0              | 0.2464     | 0.00025 | 0.6119 | 0.7640  | 0.8062 | 137.73            | 2.30                |
| With SWOP (Numpy) | No         | 0.0              | —          | —       | 0.6078 | 0.8656  | 0.9062 | 159.61            | 2.66                |
| With SWOP (TF)    | No         | 0.0              | —          | —       | 0.6076 | 0.8635  | 0.9045 | 158.35            | 2.64                |
| With SWOP         | Yes        | 0.1              | 38389.3765 | 0.00025 | 0.5825 | 0.7808  | 0.8256 | 165.30            | 2.76                |
| With SWOP         | Yes        | 0.5              | 10.3322    | 0.00025 | 0.6426 | 0.7451  | 0.7889 | 161.66            | 2.69                |
| With SWOP         | Yes        | 1.0              | 1.2792     | 0.00025 | 0.6318 | 0.7506  | 0.7991 | 159.20            | 2.65                |
| With SWOP         | Yes        | 3.0              | 0.2464     | 0.00025 | 0.5781 | 0.7972  | 0.8436 | 161.19            | 2.69                |

---


