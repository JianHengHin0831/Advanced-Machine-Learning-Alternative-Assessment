# Advanced Machine Learning – Alternative Assessment

## From Discriminative Baselines to Large Multimodal Models: A Comparative Study of CNN-BERT, BLIP-2, and Qwen3-VL for Medical Visual Question Answering

**Group Name:** Artificial Unintelligence
**Course:** WOA7015 Advanced Machine Learning (Semester 1 2025/2026)
**Supervisor:** Prof. Dr. Loo Chu Kiong

### Group Members
- Hin Jian Heng (25069335)
- Tee Chee Hong (25069322)

---

### Overview
This repository contains the source code and final report for our comparative study on Medical Visual Question Answering (Med-VQA). The project evaluates the technological shift across three distinct architectural paradigms to identify their effectiveness and limitations in interpreting radiological images (CT, MRI, X-ray).

### Repository Structure
- `med-vqa-baseline.ipynb`: Implementation and training of the discriminative baseline (ResNet-50 + BERT). Treats VQA as a fixed-vocabulary classification task.
- `med-vqa-blip-2.ipynb`: Implementation of the generative Vision-Language Model (BLIP-2). Fine-tuned using LoRA.
- `vqa-rad-qwen-3.ipynb`: Implementation of the instruction-tuned Large Vision-Language Model (Qwen-VL). Utilizes dynamic resolution and QLoRA (Int4).
- `AML AA Report.pdf`: The complete final project report detailing methodology, training dynamics, confusion matrices, and qualitative failure analysis.

### Dataset
The models are trained and evaluated on the **VQA-RAD dataset**, which contains 315 radiological images and 3,515 Question-Answer pairs manually verified by board-certified radiologists. The data features high-variance aspect ratios and resolutions, presenting a direct challenge to standard resizing pipelines.

### Key Findings
1. **The Resolution Trap:** The generative BLIP-2 model (57.76%) underperformed the simpler discriminative baseline (59.48%) on closed-ended tasks. This was caused by the fixed 224x224 image compression, which acts as a lossy algorithm that destroys subtle high-frequency visual evidence required for clinical diagnosis.
2. **Dynamic Resolution:** Qwen-VL overcame this bottleneck by processing images at their native aspect ratios and higher resolutions, achieving a dominant 77.59% accuracy on closed-ended tasks and 32.98% on open-ended tasks. 
3. **Hallucination Evolution:** While the discriminative baseline suffers from a "semantic void" (failing on open-ended queries) and BLIP-2 suffers from object hallucinations (misidentifying organs), Qwen-VL introduces "plausible hallucinations" where it identifies the correct symptoms but generates a medically coherent yet incorrect differential diagnosis (e.g., confusing Appendicitis with Diverticulitis).
4. **Accuracy-Latency Trade-off:** The high performance of Qwen-VL comes at an inference cost, operating roughly 28% slower than the lower-resolution models, which is a key consideration for real-time triage deployment.

### Usage
The experiments are containerized within standalone Jupyter Notebooks. To reproduce the results:
1. Ensure you have an environment with PyTorch and GPU support (a single NVIDIA T4 16GB VRAM was used for this project).
2. Install the necessary dependencies (transformers, peft, bitsandbytes, qwen-vl-utils) as outlined in the initial cells of the notebooks.
3. Run the notebooks sequentially to load the VQA-RAD dataset, preprocess the data, and initiate the training/evaluation loops.
