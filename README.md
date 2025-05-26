# 🐘 AI for Elephant Conservation: Smart Highlight Detection with HL-CLIP 🎞️

**Unlocking insights from countless hours of wildlife footage to protect our gentle giants.**

## The Challenge: A Needle in a Haystack 🌾

African and Asian elephants face unprecedented threats, from poaching to habitat loss. Researchers and conservationists deploy countless camera traps and drones, generating terabytes of video data. Manually sifting through this footage to find crucial moments – signs of distress, poaching activity, unique behaviors, or elephant census data – is an overwhelming, time-consuming, and costly task. Critical information can be missed, delaying vital conservation actions.

## Our Solution: Your AI Digital Ranger 💡

This project introduces an AI-powered system to automatically detect highlight moments in elephant video footage. Leveraging the cutting-edge **HL-CLIP (Highlight-CLIP)** model, we fine-tune a powerful pre-trained vision-language model to understand and identify significant events within videos of elephants. This "AI Digital Ranger" aims to:

*   **Accelerate Research:** Drastically reduce manual review time, allowing experts to focus on analysis and action.
*   **Enhance Monitoring:** Enable near real-time alerts for critical events (e.g., potential poaching).
*   **Improve Data Collection:** Efficiently catalog behaviors and occurrences for comprehensive ecological studies.
*   **Optimize Resources:** Make conservation efforts more efficient and cost-effective.

We've gone through a comprehensive process from data preparation and model training to evaluation and optimization, focusing on creating a practical and environmentally conscious AI solution.

---

## 🏆 Hackathon Submission Checklist & Details

This project fulfills the requirements for the AI for Elephants Hackathon:

### 1. 🌐 All Projects Must Be Open-Source

*   **Repository:** This entire project, including all code, model weights (for the fine-tuned versions), and documentation, is publicly available at: `[LINK_TO_YOUR_GITHUB_REPO]`
*   **License:** This project is licensed under the MIT License (see `LICENSE` file).

### 2. 🧠 AI Model/System Prototypes

*   **Model Submission:**
    *   The fine-tuned HL-CLIP model (`best_hlclip_model.pth`) and the pruned fine-tuned HL-CLIP model (`best_prune_hlclip_model.pth`) are available in the repository [e.g., in a `models/` directory or via a release].
*   **Model Card:** This `README.md` serves as the primary model card.
    *   **Model Structure & Methodology:**
        *   **Base Model:** OpenAI's CLIP (ViT-B/32 architecture).
        *   **HL-CLIP Adaptation:** We adopted the principles of HL-CLIP by fine-tuning the CLIP visual encoder. Specifically, the last 2 transformer layers of the visual encoder were unfrozen and trained, while the rest of the pre-trained weights were kept frozen. A custom classification head (comprising a Linear layer, ReLU activation, and an output Linear layer with Sigmoid) was added to predict frame-level highlight scores.
        *   **Intended Use Case:** To automatically identify and flag semantically significant segments (highlights) in videos featuring elephants. This can assist researchers in quickly locating footage of interest for behavioral studies, poaching incident detection, population monitoring, or other conservation-relevant activities. The model outputs per-frame highlight probabilities, which can then be post-processed to define highlight segments.

### 3. 🎬 Video Demonstration

*   A short, public-friendly video explaining our solution, its goals, and potential impact on elephant conservation is available here:
    `[LINK_TO_YOUR_YOUTUBE/VIMEO_VIDEO_DEMO]`
    *(The video should briefly cover the problem, how HL-CLIP works conceptually, show some example predictions, and discuss the conservation benefits.)*

### 4. 📜 Scientific Report

*   A document covering the project’s motivation, detailed methodology, dataset description, training procedures, evaluation results (including comparisons), and discussion can be found here:
    `[LINK_TO_YOUR_SCIENTIFIC_REPORT_PDF_OR_DOC]`
    *(This report would expand significantly on the details presented in this README.)*

### 5. 🌍 AI Model Footprint on Environment

We are committed to responsible AI development and have taken several steps to consider and minimize the environmental footprint of our model:

*   **1. Leveraging Pre-trained Models (Transfer Learning):**
    *   **Effort:** Our core approach utilizes OpenAI's pre-trained CLIP model. This saves an immense amount of computational resources and associated carbon emissions that would have been required to train such a large vision-language model from scratch. We only fine-tuned a small portion of the network.
    *   **Impact:** This is the single most significant factor in reducing our training carbon footprint.

*   **2. Model Pruning:**
    *   **Effort:** We implemented L1 Unstructured global pruning on the fine-tuned HL-CLIP model, targeting a 20% reduction in weights across linear and convolutional layers, followed by a short re-fine-tuning phase.
    *   **Impact:** Pruning reduces the model's parameter count, which can lead to:
        *   Smaller model size on disk (less storage energy).
        *   Potentially faster inference (fewer operations), thus lower inference energy consumption per video. Our results show that pruning maintained, and even slightly improved, the F1-score.
    *   **Details:** Pruning targeted 39 layers, including the custom head and unfrozen CLIP layers.

*   **3. PyTorch JIT Compilation (TorchScript):**
    *   **Effort:** The pruned, fine-tuned model was converted to TorchScript using `torch.jit.trace`.
    *   **Impact:** TorchScript can optimize the model graph and enable execution in more efficient C++ runtimes. This primarily benefits inference speed and can reduce CPU/GPU utilization per inference, thereby lowering energy consumption during deployment.

*   **4. Efficient Training & Development Practices:**
    *   **Dataset Subsampling:** During initial development and hyperparameter exploration, we worked with a smaller subset of the full dataset (`num_train_samples = 10 good + 10 bad videos`) to iterate quickly and avoid unnecessary computation on the full dataset until the pipeline was stable.
    *   **Limited Epochs:** We trained for a limited number of epochs (e.g., 4 epochs for initial fine-tuning and pruning re-training) sufficient to demonstrate learning and improvement, avoiding excessive training.
    *   **Colab GPU Resources:** Training was performed on Google Colab using available T4 GPUs, which are relatively power-efficient compared to some older or larger server-grade GPUs. However, precise energy consumption in a shared cloud environment is difficult to measure directly.

*   **5. Consideration of Further Quantization (Future Work for Carbon Reduction):**
    *   **Effort Explored:** We initiated steps to explore quantization using Torch-TensorRT (as seen by the `!pip install torch-tensorrt` in the notebook). While full TensorRT integration and INT8 quantization were not completed within this phase due to setup complexities (e.g., `ModuleNotFoundError` during the session), it represents a clear path for further significant footprint reduction.
    *   **Potential Impact:** Successful INT8 quantization via TensorRT or PyTorch's native static/dynamic quantization can drastically reduce model size and inference energy, especially on compatible hardware. This remains a key next step for a production-ready, low-carbon deployment.

*   **Documentation on Energy Use (Qualitative & Efforts-Based):**
    *   Direct, precise energy measurement (kWh) is challenging in a shared cloud environment like Google Colab without specialized tools.
    *   Our sustainability efforts focused on *reducing computational load* through the techniques listed above (transfer learning, pruning, JIT, efficient development).
    *   The reduction in FLOPs and parameters from pruning, and the potential for faster execution from JIT, are indirect indicators of reduced energy needs per inference compared to a non-optimized, full-precision model.
    *   **Future Steps for Quantification:** If deployed, tools like CodeCarbon could be integrated, or power measurements could be taken on dedicated hardware to quantify energy savings more precisely.

---

## 🛠️ How We Did It: The Journey

### 1. Data Preparation & Preprocessing

*   **Dataset Source:** Videos were sourced from Google Drive (`/content/drive/MyDrive/elephant_clip/`). We used separate `train_augmented` and `val` directories.
*   **Annotation Strategy:**
    *   A simple yet effective annotation strategy was used: videos placed in a subdirectory containing `/bad/` in its path (e.g., `val/bad/video.mp4`) were treated as having no specific highlight segments (empty annotation list `[]` for their `video_id`).
    *   Videos in other paths (e.g., `val/good/video.mp4` or top-level in `train_augmented`) for which no explicit annotation was provided in `annotations_data` were considered as **entirely highlight footage**. This allowed us to quickly bootstrap a dataset with positive examples.
    *   For training, we initially sampled `10` "good" (assumed full highlight) and `10` "bad" (no highlight) videos to create a balanced small dataset for faster iteration.
*   **Frame Extraction:** Videos were processed using OpenCV (`cv2.VideoCapture`). Frames were extracted at a specified `stride`.
*   **Preprocessing for CLIP:** Each extracted frame (as a PIL Image) was passed through the standard `preprocess` function provided by OpenAI's CLIP model, which includes resizing to 224x224, normalization, and conversion to a PyTorch tensor.

### 2. Model Architecture: HL-CLIP

*   **Base:** We started with the powerful ViT-B/32 (Vision Transformer) visual encoder from OpenAI's CLIP model, pre-trained on a massive dataset of image-text pairs.
*   **Fine-tuning Strategy (HL-CLIP principle):**
    *   To adapt CLIP for highlight detection, we kept most of the visual encoder frozen.
    *   Crucially, the **last 2 transformer `resblocks`** of the visual encoder were unfrozen, allowing them to learn features more specific to elephant video highlights.
    *   A **custom classification head** was added:
        *   `Linear(visual_output_dim, hidden_dim=512)`
        *   `ReLU()`
        *   `Linear(hidden_dim=512, num_classes=1)` (for binary highlight score)
        *   `Sigmoid()` to output frame-level highlight probabilities (0 to 1).

### 3. Training

*   **Device:** Training was performed on a GPU (NVIDIA T4 on Google Colab).
*   **Optimizer:** `torch.optim.AdamW` with a learning rate of `1e-5`, applied only to the parameters that require gradients (the unfrozen CLIP layers and the custom head).
*   **Loss Function:** `torch.nn.BCELoss` (Binary Cross-Entropy Loss) for the frame-level binary classification task (highlight vs. non-highlight).
*   **Data Loading:**
    *   Custom `ElephantHighlightDataset` was implemented to handle video loading, frame extraction, annotation processing, and on-the-fly transformation.
    *   `torch.utils.data.DataLoader` was used with a custom `collate_fn` to batch sequences of frames and pad them to uniform length within each batch.
*   **Epochs:** Initial fine-tuning and pruning re-training were done for 4 epochs each, monitoring validation performance.

### 4. Evaluation & Results

We performed several evaluation steps:

*   **Validation Set:** A separate set of videos (`val_video_paths`, `val_annotations_data`) was used.
*   **Metrics:** Frame-level Precision, Recall, F1-Score, and Accuracy were calculated.
*   **Optimal Threshold Finding:** For both Zero-Shot CLIP and our Fine-tuned models, we iterated through a range of possible thresholds (0.01 to 0.99, or min/max scores) to find the one that maximized the F1-score on the validation set. This ensures a fair comparison by adapting to each model's output score distribution.

**Key Results Summary:**

| Model                          | Optimal Threshold | Precision | Recall | F1-Score | Accuracy | TN | FP | FN | TP |
| :----------------------------- | :---------------: | :-------: | :----: | :------: | :------: | :-: | :-: | :-: | :-: |
| Zero-Shot CLIP                 |      0.2329       |  0.3191   | 1.0000 |  0.4839  |  0.3333  |  2  | 64 |  0  | 30 |
| Fine-tuned HL-CLIP             |      0.5049       |  0.3913   | 0.9000 |  0.5455  |  0.5312  | 24  | 42 |  3  | 27 |
| Pruned Fine-tuned HL-CLIP      |      0.4951       |  0.4590   | 0.9333 |  0.6154  |  0.6354  | 33  | 33 |  2  | 28 |

*(Note: TN/FP/FN/TP are from the validation run with the optimal threshold).*

**Observations:**

*   **Fine-tuning Works:** The fine-tuned HL-CLIP model (F1: 0.5455) significantly outperformed the Zero-Shot CLIP approach (F1: 0.4839) in terms of F1-score and overall balance of precision/recall, demonstrating the value of domain-specific adaptation.
*   **Pruning is Effective:** After pruning 20% of the weights from the fine-tuned model and re-training for a few epochs, the Pruned HL-CLIP model achieved the **highest F1-score (0.6154)**. This suggests that pruning not only reduced model complexity but also potentially acted as a regularizer, improving generalization.
*   **Zero-Shot Challenges:** Zero-Shot CLIP, while powerful for general vision-language tasks, struggled to achieve high precision for this specific highlight detection task with generic prompts, tending to recall many frames as highlights (Recall: 1.0000) but with many false positives. The optimal threshold found (0.2329) is also indicative of a very different score distribution compared to sigmoid outputs.
*   **JIT Compilation:** The pruned model was successfully traced using `torch.jit.trace` for potential inference optimization, though its direct impact on metrics wasn't re-evaluated in this phase (it primarily affects inference speed/deployability).

*(You can add plots of score distributions here if you have them from your `matplotlib` code).*

### 5. Optimization Techniques Applied

*   **Pruning:** As detailed above, L1 Unstructured global pruning was applied, followed by re-training.
*   **PyTorch JIT Compilation (TorchScript):** The pruned, fine-tuned model was converted to TorchScript.
*   *(Quantization via TensorRT was explored but not fully implemented due to initial setup issues, as noted in the Environmental Footprint section).*

---

## 🚀 How to Use This Model

1.  **Clone the Repository:**
    ```bash
    git clone [LINK_TO_YOUR_GITHUB_REPO]
    cd [REPO_NAME]
    ```

2.  **Set up Environment & Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\\Scripts\\activate
    pip install -r requirements.txt 
    # (You'll need to create a requirements.txt from your notebook: !pip freeze > requirements.txt)
    # Ensure PyTorch, CLIP, OpenCV, scikit-learn, tqdm, etc., are included.
    ```

3.  **Download Model Weights:**
    *   Download `best_hlclip_model.pth` (fine-tuned) and `best_prune_hlclip_model.pth` (pruned & fine-tuned) from the repository's releases or model storage location and place them, for example, in a `models/` directory.

4.  **Prepare Your Video(s):**
    Place the elephant videos you want to analyze in a directory.

5.  **Run Prediction:**
    Use the `predict_highlights` function provided in the notebook (or a standalone script you create from it).

    ```python
    import torch
    import clip
    from PIL import Image
    import cv2
    import os
    import numpy as np

    # --- Load your chosen model (e.g., pruned model) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "ViT-B/32"

    # Load base CLIP for visual component and preprocess
    original_clip_model_base, preprocess = clip.load(clip_model_name, device="cpu", jit=False)
    original_clip_model_base = original_clip_model_base.float().to(device)
    visual_encoder_component = original_clip_model_base.visual

    # Instantiate HLCLIPModel (definition needs to be available)
    # class HLCLIPModel(torch.nn.Module): ... (your model class definition)
    
    model_path = "models/best_prune_hlclip_model.pth" # Or best_hlclip_model.pth
    model_to_predict = HLCLIPModel(visual_encoder_component).to(device)
    
    # For PyTorch >= 1.13, weights_only=True is safer if the state_dict is just weights
    # For older versions or if it's a full model pickle (not recommended), remove weights_only
    try:
        model_to_predict.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except: # Fallback for older PyTorch or if it's not just weights
        model_to_predict.load_state_dict(torch.load(model_path, map_location=device))

    model_to_predict.eval()

    # --- Define or import predict_highlights function ---
    # def predict_highlights(video_path, model, transform, device, ...): ... (your function definition)

    # --- Predict on a new video ---
    video_to_analyze = "path/to/your/elephant_video.mp4"
    
    # Use the optimal threshold found for the model you're using
    # For pruned model, optimal_threshold was ~0.4951 from your notebook output
    optimal_threshold_pruned = 0.4951 
    
    detected_segments = predict_highlights(
        video_to_analyze,
        model_to_predict,
        preprocess,
        device,
        frames_per_clip=100, # Adjust as needed
        stride=5,            # Adjust as needed
        threshold=optimal_threshold_pruned
    )

    print(f"Detected highlights in {video_to_analyze}: {detected_segments}")
    # Output will be a list of (start_time_seconds, end_time_seconds) tuples
    ```

---

## 🌱 Future Work & Enhancements

*   **Expand Dataset:** Train on a larger, more diverse dataset of elephant videos with more granular annotations for varied behaviors.
*   **Advanced Augmentations:** Implement video-specific augmentations during training.
*   **Segment-Level mAP:** Evaluate using segment-level metrics like mean Average Precision (mAP) for a more robust comparison with other video highlight detection benchmarks.
*   **Deeper Hyperparameter Tuning:** Utilize tools like Optuna or Ray Tune.
*   **Full Quantization:** Complete INT8 quantization (e.g., using PyTorch static quantization or fully enabling TensorRT) for further carbon footprint reduction and speedup on compatible edge devices.
*   **User Interface:** Develop a simple web interface or application for conservationists to easily upload videos and visualize detected highlights.
*   **Multi-Label Classification:** Extend to classify the *type* of highlight (e.g., "feeding," "social interaction," "distress").

---

## 🙏 Acknowledgements

*   OpenAI for the pre-trained CLIP model.
*   The authors of the HL-CLIP paper for the foundational methodology.
*   The organizers of the AI for Elephants Hackathon.
*   [Any data sources or individuals who helped, if applicable]
