# SegFaultLoc
Paper: Localizing Annotation Faults in Semantic Segmentation Systems via Suspiciousness Ranking

## Overview
Semantic segmentation models are widely deployed in safety-critical applications such as autonomous driving and remote sensing, where annotation quality directly affects system reliability. However, pixel-level annotations often contain semantic and geometric faults that are difficult to detect manually and can introduce systematic biases into trained models.

To address this problem, we propose **SegFaultLoc**, a unified annotation fault localization framework for semantic segmentation datasets. SegFaultLoc detects both semantic and geometric annotation faults without retraining segmentation models on corrupted data. It extracts component-level mask representations and combines supervised semantic signals with unsupervised geometric anomaly cues through a learning-based fusion strategy, producing reliable suspiciousness scores for efficient dataset inspection.

Extensive experiments on three benchmarks demonstrate that SegFaultLoc significantly outperforms existing baselines in fault ranking and early fault detection, and correcting detected faults further improves downstream segmentation performance.

![Overview](pictures/Overview.pdf)
*Overview of the SegFaultLoc framework, illustrating the complete pipeline from data processing to fault detection and manual correction.*



<img src="pictures/Extraction.pdf" width="100%"/>
*Component-level feature extraction process.*



![Detection and Ranking](pictures/Detection&Ranking.pdf)
*Dual-spectrum detection and learning-based suspiciousness aggregation.*

## Installation
```bash
pip install -r ./requirements.txt
```
Tested with: Python 3.8 (Ubuntu 18.04), PyTorch 1.13.1, CUDA 11.3

## Data Preparation
Place your segmentation dataset in a standard layout. If your layout differs, update paths in `data/fault_injector.py` or pass custom paths to `fault_detection.py`.

Datasets used in the paper: PASCAL VOC 2012, Cityscapes, ADE20K, and COCO-Stuff.

PASCAL VOC 2012: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/

Cityscapes: https://www.cityscapes-dataset.com/

ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/

COCO-Stuff: https://github.com/nightrome/cocostuff

### Example layout
```
<DATASET_ROOT>/
├── <IMAGE_DIR>/
├── <MASK_DIR>/
└── <SPLIT_DIR>/
    ├── train.txt
    └── val.txt
```

### Fault injection
```bash
python data/fault_injector.py --mode segmentation --dataset <DATASET> --split <train|val>
```
Output: `./data/fault_annotations/{DATASET}_{split}_seg_faults_voc_style.json` (target 40% fault rate)

Note: `--fault_ratio` only affects behavior when set to 0 (generates clean data with no faults). Other values are ignored in the default image-level injection.

### Input JSON format (example)
```json
{
  "image_name": "2007_000027.jpg",
  "image_size": [500, 375],
  "labels": 15,
  "bbox": [104, 78, 375, 183],
  "area": 12543,
  "fault_type": 0,
  "mask_path": "./dataset/<DATASET_PATH>/SegmentationClass/2007_000027.png"
}
```

## Quick Start (full workflow)
```bash
python fault_detection.py --mode full \
  --dataset <DATASET> \
  --set <train|val> \
  --num_classes <NUM_CLASSES> \
  --img_root <IMAGE_ROOT> \
  --json_path ./data/fault_annotations/<DATASET>_<set>_seg_faults_voc_style.json \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --loss_type gce
```


## Full Workflow
### 1) Extract embeddings
```bash
python fault_detection.py --mode extract \
  --dataset <DATASET> --set <train|val> \
  --img_root <IMAGE_ROOT> \
  --json_path ./data/fault_annotations/<DATASET>_<set>_seg_faults_voc_style.json \
  --batch_size 64
```
Output: `./data/embeddings/{DATASET}{SET}_embeddings.pkl`

### 2) Train models
```bash
python fault_detection.py --mode train \
  --dataset <DATASET> --set train \
  --num_classes <NUM_CLASSES> \
  --embeddings_path ./data/embeddings/{DATASET}Train_embeddings.pkl \
  --batch_size 64 --learning_rate 1e-3 --loss_type gce
```
Outputs:
- `./models/embedding_classifier_{DATASET}_train.pth`
- `./models/egad_{DATASET}_train.pth`
- `./models/egad_{DATASET}_train_geo_stats.npz`
- `./models/ranking_network_{DATASET}_train.pth`

### 3) Inference
```bash
python fault_detection.py --mode infer \
  --dataset <DATASET> --set <val|test> \
  --num_classes <NUM_CLASSES> \
  --embeddings_path ./data/embeddings/{DATASET}{SET}_embeddings.pkl \
  --batch_size 64
```
Outputs:
- `./{dataset}_semantic_results.pkl`
- `./{dataset}_semantic_results.json`
- `./embedding_suspiciousness_scores_{DATASET}_{set}.json`

### 4) Evaluate
```bash
python fault_detection.py --mode evaluate --dataset <DATASET> --set <val|test> --num_classes <NUM_CLASSES>
```
Output: `./embedding_metrics_{DATASET}_{set}.json`

## Parameters
| Arg | Description |
| --- | --- |
| `--mode` | Workflow stage selector |
| `--dataset` | Dataset name |
| `--set` | Dataset split name |
| `--num_classes` | Number of semantic classes |
| `--batch_size` | Batch size for extract/train/infer |
| `--loss_type` | Training loss type |
| `--epochs` | Override training epochs |

## Outputs
- Embeddings: `./data/embeddings/{DATASET}{SET}_embeddings.pkl`
- Models: `./models/embedding_classifier_{DATASET}_train.pth`, `./models/egad_{DATASET}_train.pth`, `./models/ranking_network_{DATASET}_train.pth`
- Inference: `./{dataset}_semantic_results.pkl/json`, `./embedding_suspiciousness_scores_{DATASET}_{set}.json`
- Metrics: `./embedding_metrics_{DATASET}_{set}.json`

## Results

SegFaultLoc vs. SegFormer & FD results (FD = Focal+Dice).

<table border="1">
  <tr>
    <th>Split</th>
    <th>Dataset</th>
    <th>APFD (SegFaultLoc / SegFormer &amp; FD)</th>
    <th>Recall@40 (SegFaultLoc / SegFormer &amp; FD)</th>
    <th>Precision@10 (SegFaultLoc / SegFormer &amp; FD)</th>
    <th>EXAM (SegFaultLoc / SegFormer &amp; FD)</th>
    <th>nDCG@10 (SegFaultLoc / SegFormer &amp; FD)</th>
    <th>MAP (SegFaultLoc / SegFormer &amp; FD)</th>
  </tr>

  <tr>
    <td rowspan="3">Test</td>
    <td>VOC</td>
    <td>72.40 / 63.24</td>
    <td>75.97 / 52.30</td>
    <td>95.16 / 77.72</td>
    <td>0.544 / 0.585</td>
    <td>95.50 / 78.29</td>
    <td>0.947 / 0.891</td>
  </tr>
  <tr>
    <td>Cityscapes</td>
    <td>70.59 / 63.80</td>
    <td>73.89 / 59.78</td>
    <td>96.54 / 74.65</td>
    <td>0.027 / 0.030</td>
    <td>96.91 / 74.54</td>
    <td>0.850 / 0.723</td>
  </tr>
  <tr>
    <td>ADE20K</td>
    <td>67.16 / 57.96</td>
    <td>65.17 / 48.32</td>
    <td>91.67 / 73.75</td>
    <td>0.131 / 0.171</td>
    <td>92.73 / 74.41</td>
    <td>0.818 / 0.715</td>
  </tr>
</table>

SegFormer accuracy (Acc/mIoU) under different training and testing configurations.

<table border="1">
  <tr>
    <th colspan="5">VOC (Test Set Acc/mIoU)</th>
  </tr>
  <tr>
    <th>Train Config</th>
    <th>Dirty</th>
    <th>Baseline (SegFormer &amp; FD)</th>
    <th>Ours</th>
    <th>Clean</th>
  </tr>
  <tr>
    <td>Dirty</td>
    <td>0.621/0.472</td>
    <td>0.770/0.647</td>
    <td>0.821/0.695</td>
    <td>0.826/0.702</td>
  </tr>
  <tr>
    <td>Baseline (SegFormer &amp; FD)</td>
    <td>0.640/0.487</td>
    <td>0.792/0.669</td>
    <td>0.849/0.729</td>
    <td>0.855/0.736</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>0.650/0.498</td>
    <td>0.810/0.687</td>
    <td>0.872/0.747</td>
    <td>0.882/0.760</td>
  </tr>
  <tr>
    <td>Clean</td>
    <td>0.655/0.502</td>
    <td>0.815/0.694</td>
    <td>0.876/0.749</td>
    <td>0.887/0.765</td>
  </tr>
</table>

<table border="1">
  <tr>
    <th colspan="5">Cityscapes (Test Set Acc/mIoU)</th>
  </tr>
  <tr>
    <th>Train Config</th>
    <th>Dirty</th>
    <th>Baseline (SegFormer &amp; FD)</th>
    <th>Ours</th>
    <th>Clean</th>
  </tr>
  <tr>
    <td>Dirty</td>
    <td>0.726/0.263</td>
    <td>0.761/0.306</td>
    <td>0.866/0.412</td>
    <td>0.880/0.426</td>
  </tr>
  <tr>
    <td>Baseline (SegFormer &amp; FD)</td>
    <td>0.728/0.265</td>
    <td>0.777/0.309</td>
    <td>0.876/0.421</td>
    <td>0.890/0.439</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>0.728/0.266</td>
    <td>0.788/0.314</td>
    <td>0.882/0.442</td>
    <td>0.890/0.453</td>
  </tr>
  <tr>
    <td>Clean</td>
    <td>0.740/0.294</td>
    <td>0.792/0.348</td>
    <td>0.894/0.488</td>
    <td>0.909/0.506</td>
  </tr>
</table>

<table border="1">
  <tr>
    <th colspan="5">ADE20K (Test Set Acc/mIoU)</th>
  </tr>
  <tr>
    <th>Train Config</th>
    <th>Dirty</th>
    <th>Baseline (SegFormer &amp; FD)</th>
    <th>Ours</th>
    <th>Clean</th>
  </tr>
  <tr>
    <td>Dirty</td>
    <td>0.550/0.170</td>
    <td>0.588/0.226</td>
    <td>0.648/0.257</td>
    <td>0.689/0.300</td>
  </tr>
  <tr>
    <td>Baseline (SegFormer &amp; FD)</td>
    <td>0.519/0.156</td>
    <td>0.591/0.229</td>
    <td>0.669/0.282</td>
    <td>0.694/0.324</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>0.516/0.152</td>
    <td>0.650/0.242</td>
    <td>0.701/0.292</td>
    <td>0.720/0.342</td>
  </tr>
  <tr>
    <td>Clean</td>
    <td>0.598/0.203</td>
    <td>0.682/0.263</td>
    <td>0.714/0.307</td>
    <td>0.759/0.384</td>
  </tr>
</table>
