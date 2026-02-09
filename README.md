# Multi-Method Deepfake Detection (Spatial + MRI-GAN + Temporal + Fusion)

This repository provides an **end-to-end, production-oriented deepfake detection toolkit** that combines multiple complementary signals:

- **Spatial frame-level CNN** (EfficientNet) for texture and blending artifacts  
- **Perceptual MRI-based features** using an MRI-GAN-style generator  
- **Temporal modeling** using a 1D CNN over frame embeddings to detect flicker and motion inconsistencies  
- **Confidence-based adaptive ensemble fusion** for robust final decisions  
- **FastAPI Web UI** for interactive video upload and inference  

The goal is to move beyond single-model detectors and build a **robust, interpretable, and deployable** deepfake forensics pipeline.

---

##  What makes this repo different

This codebase is inspired by the **MRI-GAN** research idea and extends it into a **complete, runnable system** with:

- An **end-to-end video pipeline** (face detection → preprocessing → inference → fusion)
- Multiple **complementary detection branches** (spatial, perceptual, temporal)
- **Confidence-based adaptive fusion** instead of simple averaging
- **Optional temperature scaling** for probability calibration
- A **FastAPI web interface** and CLI tools for practical usage
- A “batteries-included” project structure for research and experimentation

**Reference paper (MRI-GAN):** https://arxiv.org/abs/2203.00108

---

##  System Architecture (High-Level Flow)

1. **Input Video**
2. **Face Detection & Preprocessing**
   - Detect faces using MTCNN
   - Align and crop faces to fixed resolution (e.g., 256×256)
   - Sample frames from the video
3. **Parallel Detection Branches**
   - (A) Spatial Frame-Level Model (EfficientNet)
   - (B) Perceptual MRI-GAN-Based Model
   - (C) Temporal Model (1D CNN over frame embeddings)
4. **Probability Calibration (Optional)**
   - Apply **Temperature Scaling** to logits for better-calibrated confidence scores
5. **Confidence-Based Adaptive Ensemble Fusion**
   - Compute confidence for each branch
   - Dynamically weight each model’s prediction
   - Produce a final fused probability
6. **Final Output**
   - Real / Fake prediction + confidence score
7. **Web UI / CLI**
   - Results shown via FastAPI UI or printed via CLI

---

## Model Details

### 1️⃣ Spatial Frame-Level Model (EfficientNet)

**Purpose:**  
Detects **spatial artifacts** within individual frames such as:
- Texture inconsistencies  
- Blending errors around face boundaries  
- Unnatural edges, lighting, or shading  
- GAN-specific visual fingerprints  

**How it works:**  
Each face frame is passed through **EfficientNet-B0** as a feature encoder. Feature maps are reduced using **Global Average Pooling**, and a fully connected layer with sigmoid outputs the **probability that the frame is fake**.

**Pros:** Fast, accurate, strong baseline.  
**Cons:** Ignores temporal information.

---

### 2️⃣ Perceptual Model (MRI-GAN + CNN Classifier)

## MRI-GAN concept (high-level)

MRI-GAN generates an “MRI map” for an input face frame. For fake frames, the map tends to highlight synthesized regions; for real frames it tends to be near-black.

![MRI-GAN architecture](images/mri_model_arch.png)

### Training visuals

![Discriminator](images/dis_model.png)
![Generator](images/gen_model.png)

![MRI dataset formulation](images/mri_df_dataset_gen.png)

![MRI sample output](images/MRI_demo.png)

---

### 3️⃣ Temporal Model (1D CNN over Frame Embeddings)

**Purpose:**  
Detects **temporal inconsistencies** such as:
- Flickering artifacts  
- Unstable textures over time  
- Unnatural facial motion  
- Warping or jitter across frames  

**How it works:**  
Each frame is passed through the EfficientNet encoder to extract a **feature embedding**. Embeddings from multiple frames are stacked into a sequence and passed through a **1D CNN** over the time dimension to learn short-term temporal patterns. After pooling and classification, the model outputs the **probability that the video is fake based on temporal behavior**.

**Pros:** Captures motion-based artifacts missed by frame-only models, lightweight and efficient.  
**Cons:** Requires multiple frames, focuses mainly on short-term temporal dependencies.

---

##  Probability Calibration (Temperature Scaling)

Neural networks are often **overconfident** in their predictions. We optionally apply **Temperature Scaling** as a post-hoc calibration method:

- Logits are divided by a learned temperature **T**
- Sigmoid is applied after scaling
- This **does not change accuracy**
- But makes probability scores **more reliable and better calibrated**

This is important because the system uses **confidence-based ensemble fusion**.

---

##  Confidence-Based Adaptive Ensemble Fusion

Each branch (Spatial, MRI, Temporal) outputs a calibrated probability and a confidence score based on distance from uncertainty (0.5).

**Fusion process:**
1. Compute confidence for each model  
2. Convert confidence to **adaptive weights**  
3. Compute a **weighted average** of predictions  
4. Produce the final fused probability  

This allows the system to **trust the most reliable model for each video** instead of using fixed or equal weights.

---
## Quickstart (inference)

### 1) Install

- If you use conda: `conda env create -f environment.yml`
- Or use pip: `pip install -r requirements.txt`

### 2) Configure

Edit `config.yml` (or `config_windows.yml` on Windows) to point to your local dataset / cache paths.

### 3) Run CLI inference

Use the predictor entrypoint in `deep_fake_detect_app.py` (the web server calls this too).

- Example:
  - `python deep_fake_detect_app.py --input_videofile <path> --method plain_frames`
  - `python deep_fake_detect_app.py --input_videofile <path> --method MRI`
  - `python deep_fake_detect_app.py --input_videofile <path> --method fusion`
  - `python deep_fake_detect_app.py --input_videofile <path> --method temporal`

---

## Web UI (FastAPI)

The web app lives under `web/` and provides a simple upload + prediction interface.

- Install web deps: `pip install -r web/requirements.txt`
- Run server: `uvicorn web.server:app --reload --port 8000`
- Open: http://127.0.0.1:8000

---

## Datasets & large files

Datasets, generated artifacts, and model weights are intentionally excluded from git (see `.gitignore`).
Use `download_models.py` or your own storage to manage weights.

---

## Citation / Credits

If you use MRI-GAN ideas academically, cite the original paper:

```
Pratikkumar Prajapati and Chris Pollett, MRI-GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment. arXiv preprint arXiv:2203.00108 (2022)
```

This repository started from the public implementation at https://github.com/pratikpv/mri_gan_deepfake and was extended with additional pipelines and a web interface.
