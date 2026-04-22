
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

## 📦 Checkpoints

Due to space limitations, we host our SUTs and finetuned generator checkpoints on Google Drive. 
**[https://drive.google.com/drive/folders/1naXGbftzFZoioL32BAz2na5VoN2ea-mt?usp=drive_link]**

Please download the folders and merge them into the following local directories `./local_models/`

> **Note:** These finetuned generators were trained using the [official StyleGAN2-ADA codebase](https://github.com/NVlabs/stylegan2-ada-pytorch) and pre-trained checkpoints. The fine-tuning process was conducted with the following hyperparameters:
> ```bash
> --snap=5 --aug=ada --target=0.7 --freezed=10 --batch=8 --gamma=1 --glr=0.0001 --dlr=0.0015 --kimg=100
> ```
## 📝 Citation

A preprint of the paper can be found on [arXiv](https://arxiv.org/abs/2503.07222).

If you use our work in your research, or it helps it, or if you simply like it, please cite it in your publications. 
Here is an example BibTeX entry:

```
@article{chen2026feature,
  title={Feature-Aware Test Generation for Deep Learning Models},
  author={Chen, Xingcheng and Weissl, Oliver and Stocco, Andrea},
  journal={arXiv preprint arXiv:2601.14081},
  year={2026}
}
```
---


