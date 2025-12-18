
# DETECT: Feature-Aware Test Generation for Deep Learning Models

This repository provides a unified pipeline for generating test cases in vision models using disentangled latent space manipulations based on pretrained StyleGAN generators.



All tasks are executed via a unified entry point in `main.py`, which supports different configurations of perturbation and oracle strategies.

---

## 🔧 Requirements

* Python 3.9.19
* PyTorch (CUDA supported)
* Other dependencies listed in `requirements.txt`

Make sure to download and place the pretrained models (GANs, classifiers, and segmentation models) at the paths defined in `configs.py`.

---

## 🧭 Tasks and Supported Models

| Task   | Dataset  | Classifier          |
| ------ | -------- | ------------------- |
| facial | CelebA   | ResNet50 / SWAG ViT | 
| dog    | LSUN Dog | ReXNet-150          | 
| yolo   | LSUN Car | YOLOv8n             | 

---

## 🚀 Usage

Run the main script with the desired configuration:

```bash
python main.py --task facial --model small --config smoothgrad --oracle confidence_drop
```

### Common Arguments

| Argument                 | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| `--task`                 | Task to run: `facial`, `dog` , `yolo`                         |
| `--model`                | `small` or `large` model (only for facial task)               |
| `--config`               | Attribution method: `gradient`, `smoothgrad`, `occlusion`     |
| `--oracle`               | Oracle strategy: `confidence_drop`, `misclassification`       |
| `--extent_factor`        | Perturbation strength (default: 10; 20 for misclassification) |
| `--truncation_psi`       | Truncation value for StyleGAN (0.7 for facial, 0.5 for yolo)  |
| `--confidence_threshold` | Threshold for confidence drop (e.g., 0.4)                     |
| `--target_logit`         | Target logit index (e.g., 15 for glasses attribute)           |
| `--start_seed`           | Starting random seed (default: 0)                             |
| `--end_seed`             | Ending random seed (exclusive)                                |

Example:

```bash
python main.py --task yolo --config smoothgrad --oracle misclassification --start_seed 10 --end_seed 50
```

---

## 📁 Output

Results will be saved under:

```
generate_image_base_dir/
└── runs_/
    ├── [model]_[config]_[oracle]/
    │   ├── [target_logit]/ (for facial)
    │   └── [seed_id]/ (for dog/yolo)
```

Each folder contains:

* Original and perturbed images
* Prediction logs
* Perturbation metadata

## 📦 Checkpoints and Training Code

Due to space limitations, we only include the **inference pipeline** in this repository. The code used to **train or fine-tune** the following models is available in a separate archive:

* StyleGAN2 generators (with limited fine-tuning via ADA)
* Classifiers (ResNet50, SWAG-ViT, ReXNet, YOLOv8)

We provide **trained or fine-tuned checkpoints upon request**. Please contact the authors or repository maintainers if you would like access to specific models.

## 🔍 Notes

---

## 📝 Citation



---


